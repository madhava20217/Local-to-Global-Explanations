# Local2Global: Obtaining Global Importances from Local Scores

## Description

This project is about deriving global interpretations from local perspectives. There exist few approaches to obtain a global perspective from multiple local points, and the ones that exist, usually involve a simple aggregation without accounting for the distribution.
The method proposed in this work requires a sample distribution of data points, on which a Gaussian mixture model is fit primarily because of their ability to handle multimodal data (data with multiple modes in the distribution). Points are sampled from the distribution, and local interpretations for the samples computed using LIME or SHAP are aggregated using Monte Carlo trials.
A second method is also implemented in the repository, involving importance sampling from a standard normal distribution, leading to a smaller variance in results.
The primary evaluation metric proposed in this work involves benchmarking the net accuracy drop on successive elimination of features reported by Global SHAP, Local2Global-MC (Monte Carlo) and Local2Global-IS (importance sampling). The more important the feature eliminated, the larger the drop is accuracy is the premise for this testing methodology. A decision tree is used, which typically overfits the data. Because of this, any features that make it impossible to overfit should result in a significant drop in performance.
The refitting on the reduced feature space happens 100 times to reduce the variance of the results.

For more information, please look at the <a href = 'Local2Global.pdf'>Local2Global.pdf</a> file.
