import numpy as np

class MLE:
    def __init__(self, data, density="gaussian"):
        self.data = data
        self.density = density
        if (density != "gaussian"):
            raise Exception("The following ", density, "is not supported")

    def get_params(self):
        if self.density == 'gaussian':
            mean = self.__get_mean_gaussian()
            cov = self.__get_cov_gaussian()
            return mean, cov

    def __get_mean_gaussian(self):
        return np.mean(self.data, axis=0)

    def __get_cov_gaussian(self):
        n = self.data.shape[0]
        mean = self.__get_mean_gaussian()
        cov = np.zeros((mean.shape[0], mean.shape[0]))
        for i in range(n):
            cov += np.outer(self.data[i] - mean, self.data[i] - mean)

        return 1/n * cov