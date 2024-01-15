# Statistical-Learning-Toolkit
A comprehensive Statistical Learning Toolkit featuring a diverse set of classifiers based on Bayesian decision theory, parameter and density estimation techniques, and dimensionality reduction procedures.

## About 
This statistical learning toolkit has a wide range of functionalities. Following is a brief description on the methods that are currently supported:

### Dimensionality Reduction Techniques
* **Multiple Discriminant Analysis (MDA)**: A dimensionality reduction procedure which aims to minimize the loss of "classifiability" by keeping the lower dimensional data best seperated in the least squares sense. This method poses an additional requirement,  necessitating the availability of class labels for the provided data.
* **Principal Component Analysis (PCA)**: A dimensionality reduction procedure which aims to minimize the loss of "data fidelity", measured in the least square sense.

### Parameter Estimation Procedures
* **Maximum Likelihood Estimation (MLE)**: A method of estimating the parameters of an assumed probability distribution by maximizing the likelihood of obtaining the observed data as a function of the unknown parameters. Here, the unknown parameters are considered to be fixed, deterministic parameters. Currently, this technique is only supported for Gaussian distributions.
* **Bayesian Parameter Estimation**: Here, the unknown parameters are considered to be random variables with known/assumed prior pdfs. This approach estimates the parameter values by updating a prior belief about model parameters (i.e., prior distribution) with new evidence (i.e., observed data) via a likelihood function, resulting in a posterior distribution. An estimate of the unknown parameter in a number of ways from the posterior distribution (e.g. mean, mode, median). Currently, this technique is only supported for Gaussian distributions.

### Classifiers

### Pipelines

## Usage
