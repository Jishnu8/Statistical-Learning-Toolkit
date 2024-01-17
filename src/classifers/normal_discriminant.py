import numpy as np

class NormalDiscriminant1:
    """
    NormalDiscriminant1 class

    In this case, only the mean of the normal density and a single value for the variance are unknown. 
    The assumption is that the features are statistically independent and each feature has the same 
    variance regardless of class.
    """

    def __init__(self, mean, sigma=1, prior=None):
        """
        Initializes class based on the required arguments. 

        Args:
            mean: mean of the class conditional normal density 
            sigma: variance
            prior: prior of the given class
        """

        self.mean = mean
        self.sigma = sigma
        self.prior = prior

    def out(self, x):
        """
        Returns the output of the discriminant function for a given sample

        Args:
            x: given training sample
        """

        if self.prior==None:
            #assume all classes have the same prior, hence sigma doesn't play a role
            g = -(np.linalg.norm(x - self.mean))**2
        else:
            g = (-1/(2*self.sigma**2))*((np.linalg.norm(x - self.mean))**2) + np.log(self.prior)

        return g

class NormalDiscriminant2:
    """
    NormalDiscriminant2 class

    In this case, it is assumed that all classes have the same, but arbitrary covariance matrix. 
    Hence, only class specific means and a common covariance matrix are parameters to be estimated.
    """

    def __init__(self, mean, common_cov, prior=None):
        """
        Initializes class based on the required arguments. 

        Args:
            mean: mean of the class conditional normal density 
            common_cov: common covariance matrix 
            prior: prior of the given class
        """

        self.prior = prior
        conv_inv = np.linalg.inv(common_cov)
        self.w = np.matmul(conv_inv, mean)
        self.w0 = (-1/2)*np.matmul(np.matmul(np.transpose(mean), conv_inv), mean)

    def out(self, x):
        """
        Returns the output of the discriminant function for a given sample

        Args:
            x: given training sample
        """

        if self.prior == None:
            g = np.matmul(np.transpose(self.w), x) + self.w0
        else:
            g = np.matmul(np.transpose(self.w), x) + self.w0 + np.log(self.prior)

        return g

class NormalDiscriminant3:
    """
    NormalDiscriminant3 class

    In this case, no restriction are made. Class specific means and covariance matrices are 
    parameters to be estimated.
    """

    def __init__(self, mean, cov, prior):
        """
        Initializes class based on the required arguments. 

        Args:
            mean: mean of the class conditional normal density 
            cov: covariance matrix of the class conditional normal density  
            prior: prior of the given class
        """

        cov_inv = np.linalg.inv(cov)
        self.W = -(1/2)*cov_inv
        self.w = np.matmul(cov_inv, mean)
        cov_eig_values, cov_eig_vecs = np.linalg.eig(cov)

        
        cov_eig_values_log = np.log(cov_eig_values)
        tmp = np.sum(cov_eig_values_log)
        self.w0 = (-1/2)*np.matmul(np.matmul(np.transpose(mean), cov_inv), mean) \
              - (1/2)*tmp + np.log(prior)

    def out(self, x):
        """
        Returns the output of the discriminant function for a given sample

        Args:
            x: given training sample
        """

        g = np.matmul(np.matmul(np.transpose(x), self.W),x) + np.matmul(np.transpose(self.w), x) + self.w0

        return g
