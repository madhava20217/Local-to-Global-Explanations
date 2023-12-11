from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from lime import lime_tabular
import numpy as np
from tqdm import tqdm
from scipy.stats import uniform, norm
from scipy.special import softmax

class Local2GlobalExplainer:
    def __init__(self, x_train, model, n_classes, components = None):
        '''Initialization function for Local2GlobalExplainer
        
        Arguments:
        1. x_train: training data
        2. model: the model to run explanations for
        3. n_classes: number of classes'''
        self.model = model
        self.data = x_train
        
        # LIME model
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.data, 
            mode = 'classification'     
            )
        
        if components == None:
            components = n_classes*4
        
        # cached mcmc explanations
        self.mcmc_explanations = None
        self.mcmc_agg = None
        
        # cached importance sampling explanations
        self.imp_explanations = None
        self.imp_agg = None
        
        # fitting a gaussian mixture model to take care of multimodal data
        self.gmm = BayesianGaussianMixture(n_components = components).fit(x_train)
        
    def get_optimal_gmm(n_components, x_train):
        '''Helper function 
        
        **UNUSED**
        
        '''
        c = round(n_components)
        gmm = GaussianMixture(n_components = c).fit(x_train)
        return gmm.bic(x_train)
        
    def get_local_interpretation(self, sample):
        '''Function to get LIME interpretations for a sample
        
        Arguments:
        1. sample: the incoming sample
        2. num_features: number of features to '''
        exp = self.explainer.explain_instance(sample, self.model.predict_proba, num_features = len(sample))
        local_exp = list(exp.local_exp.values())[0]
        local_exp = sorted(local_exp)
        
        explanations = [x[1] for x in local_exp]
        return explanations
    
    def rank_explanations(self, explanations):
        '''Helper function to rank explanations sorted in the order of highest magnitude to lowest magnitude
        
        Arguments:
        1. explanations: aggregated explanations
        
        Returns:
        1. sorted list for explanations with feat indices and corresponding importances'''
        return sorted(list(zip(range(len(explanations)), explanations)), key = lambda x: -abs(x[1]))
    
    def get_only_feature_importance(self, explanations):
        '''Helper function to only get features
        
        Arguments:
        1. explanations: aggregated explanations
        
        Returns:
        1. Sorted list for explanations with just feature indices'''
        ranks = self.rank_explanations(explanations)
        return [x[0] for x in ranks]
    
    def mcmc_estimate(self, num_samples):
        '''Function to run Markov chain Monte Carlo approx based explanations. Samples q(x) from a standard normal distribution
        
        Arguments:
        1. num_samples: number of samples to sample from
        
        Returns:
        1. agg_explanations: an array containing aggregated explanations
        2. explanations: explanations for each sample'''
        samples, gmm_class = self.gmm.sample(num_samples)       # generate samples from the fit gmm
        explanations = []                                       # list to store explanations
        for sample in tqdm(samples):
            interpret = np.array(self.get_local_interpretation(sample))
            # sigmoid_interpretation = self.get_scores(interpret)
            interpret = np.abs(interpret)
            explanations.append(interpret)
            
        agg_explanations = np.mean(np.array(explanations), axis = 0)        # aggregating
        
        self.mcmc_agg = explanations
        self.agg_explanations = agg_explanations
        return agg_explanations, explanations
    
    def get_scores(self, ex):
        '''Function to normalize scores using a softmax function multiplied by correlation signs'''
        signs = np.sign(ex)
        abs_softmax = softmax(np.abs(ex))
        return abs_softmax*signs
    
    def importance_sampling(self, num_samples):
        '''Function to run importance sampling for explanations. Samples q(x) from a standard normal distribution
        
        Arguments:
        1. num_samples: number of samples to sample from
        
        Returns:
        1. agg_explanations: an array containing aggregated explanations
        2. explanations: explanations for each sample'''
        q = np.random.randn(num_samples)
        samples, gmm_class = self.gmm.sample(num_samples)       # generate samples from the fit gmm
        scores = np.exp(self.gmm.score_samples(samples))        # p(x)
        qx = norm.pdf(q)                                        # q(x)
        
        importance = scores/qx
        explanations = []
        
        for i in tqdm(range(num_samples)):
            interpret = np.array(self.get_local_interpretation(samples[i]))
            # sigmoid_interpretation = self.get_scores(interpret)
            importance_weighted_interpretation = np.abs(interpret*importance[i])
            explanations.append(importance_weighted_interpretation)
            
        agg_explanations = np.mean(np.array(explanations), axis = 0)
        self.imp_explanations = explanations
        self.imp_agg = agg_explanations
        
        return agg_explanations, explanations
        