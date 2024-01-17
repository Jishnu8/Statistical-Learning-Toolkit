import numpy as np

class BayesParamEstimation:
    """
    BayesParamEstimation class

    Bayes Param Estimation class contains all the required methods to perform Bayes Parameter Estimation
    """

    def __init__(self, data, density="gaussian", mode="MAP"):
        """
        Initializes class based on the required arguments. 

        Args:
            data: features
            dim: density assumed for the features (currently only 
            Gaussian is supported)
            mode: method by which the unknown parameters are estimated from 
            the posterior distribution (currently only MAP estimate is supported)
        """

        self.data = data
        self.density = density
        if density == "gaussian":
            self.estimator = BayesParamEstimationGaussian(self.data)
        else:
            raise Exception("The following ", density, "is not supported")

    def get_params(self, cov, prior_params):
        """
        Returns an estimate of the unknown parameters
        """
        
        return self.estimator.get_parameters(cov, prior_params)

    
class BayesParamEstimationGaussian:
    def __init__(self, data):
        self.data = data

    def get_parameters(self, params):
        cov = params[0] #known covariance matrix of Gaussian
        prior_params = params[1:3] #inital guess of mean and covariance matrix
        mean = self.__get_posterior_mean_gaussian(cov, prior_params)
        cov = self.__get_posterior_cov_gaussian(cov, prior_params)
        return mean, cov
        
    def __get_posterior_mean_gaussian(self, cov, prior_params):
        n = self.data.shape[0]
        mle_mean = self.__get_mean_gaussian()
        common_term = np.matmul(prior_params[1], np.linalg.inv(prior_params[1] + 1/n * cov))
        posterior_mean = np.matmul(common_term, mle_mean + (1/n) * prior_params[0])

        return posterior_mean

    def __get_posterior_cov_gaussian(self, cov, prior_params):
        n = self.data.shape[0]
        posterior_cov = 1/n * np.matmul(np.matmul(prior_params[1], np.linalg.inv(prior_params[1] + 1/n * cov)), cov)
        
        return posterior_cov

    def __get_mean_gaussian(self):
        return np.mean(self.data, axis=0)