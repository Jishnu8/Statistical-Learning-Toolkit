import numpy as np
from src.utils import split_classes

class MDA:
    """
    MDA class

    Multiple Discriminant Analysis class contains all the required methods to perform dimensionality reduction.
    """

    def __init__(self, data_x, data_y, dim):
        """
        Initializes class based on the required arguments. 

        Args:
            data_x: features of the data
            data_y: labels of the data
            dim: dimension to which data is projected
        """
    
        self.data_x = data_x
        self.data_y = data_y
        self.dim = dim
        self.class_separated_data_x = split_classes(self.data_x, self.data_y)
        self.mean = np.mean(data_x, axis=0).reshape(1, -1)
        self.class_means = np.array([np.mean(self.class_separated_data_x[i], axis=0) for i in range(10)])
    
    def get_S_w(self):
        """
        Returns 'within-scatter matrix'
        """

        S_w = 0
        for i in range(10):
            S_i = (self.class_separated_data_x[i] - self.class_means[i]).T.dot(self.class_separated_data_x[i] - self.class_means[i])
            S_w += S_i

        return S_w
    
    def get_S_b(self):
        """
        Returns 'between-scatter matrix'
        """

        S_b = 0
        n = np.array([self.class_separated_data_x[i].shape[0] for i in range(10)])
        diag_n = np.diag(n)
        S_b_i = ((self.class_means - self.mean).T).dot(diag_n.dot(self.class_means - self.mean))
        S_b += S_b_i

        return S_b
    
    def get_features(self):
        """
        Returns MDA reduced features and projection vectors
        """

        S_w = self.get_S_w()
        S_b = self.get_S_b()

        S = np.linalg.pinv(S_w).dot(S_b)

        eigenvalues, eigenvectors = np.linalg.eig(S)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        proj_vectors = np.real(sorted_eigenvectors[:, :self.dim])

        reduced_train_data = self.data_x.dot(proj_vectors)

        return reduced_train_data, proj_vectors
