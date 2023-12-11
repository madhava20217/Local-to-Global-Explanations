# Local2Global: Obtaining Global Importances from Local Scores

## Description

This project is about deriving global interpretations from local perspectives in a post hoc manner. There exist few approaches to obtain a global perspective from multiple local points, and the ones that exist, usually involve a simple aggregation without accounting for the distribution.
The method proposed in this work requires a sample distribution of data points, on which a Gaussian mixture model is fit primarily because of their ability to handle multimodal data (data with multiple modes in the distribution). Points are sampled from the distribution, and local interpretations for the samples computed using LIME or SHAP are aggregated using Monte Carlo trials.
A second method is also implemented in the repository, involving importance sampling from a standard normal distribution, leading to a smaller variance in results.
The primary evaluation metric proposed in this work involves benchmarking the net accuracy drop on successive elimination of features reported by Global SHAP, Local2Global-MC (Monte Carlo) and Local2Global-IS (importance sampling). The main class implemented here, 