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
* **Bayes Classifier**: When the prior and class conditional density functions are known, Bayes decision rule (which predicts the posterior probabilities) leads to minimum error-rate classification. Bayes classifier can also be easily represented as a discriminant function given by $g_i(x) = ln(P(x/ω_i) + ln(P(w_i))$. Evidently, in practical scenarios, priors and class conditional densities are rarely known. The most common technique is to assume the class conditional density as a normal distribution with unknown parameters (which are estimated using a parameter estimation technique), leading to the following 3 discriminant functions:
  
  * **Normal Discriminant 1 ($Σ_i = σ^2I. $)**: In this case, only the mean of the normal density unknown. The assumption is that the features are statistically independent and
each feature has the same variance regardless of class.

  * **Normal Discriminant 2 (Σi = Σ)**: In this case, it is assumed that all classes have the same, but arbitrary covariance matrix. Hence, only class specific means and a common covariance matrix are parameters to be estimated.
    
  * **Normal Discriminant 2 (Σi = Arbitrary)**: In this case, no restriction are made. Class specific means and covariance matrices are parameters to be estimated.

* **K-Nearest Neighbors (KNN)**: This non-parametric, supervised learning classifier assigns a class label to a data point based on the majority class among its k nearest neighbors in the feature space. Proximity is measured by some relevant distance metric in the feature space, and is most commonly the Minkowski metric.

* **PCA Approximation Method**: This is a custom method which is both extremly efficient and accurate, created and supported by this toolkit. Given the training set, it involves finding the principal components (using PCA) for each class. Given the eigenspaces associated with each class, a test sample is first projected onto the eigenspace of each class and then reconstructed back to its original dimension. The prediction is made based on the class for which this reconstruction closely aligns with the original test sample, measured by the mean square error.
  
### Pipelines
The general paradigm for most classification tasks involves dimensionality reduction, parameter estimation (if Bayes classifiers are used) and the usage of the classifier itself. For all classifiers mentioned above (except for the PCA Approximation Method which does not follow this general paradigm), this toolkit supports a single method to perform classification with the desired procedures of each part of the pipeline.

## Usage
