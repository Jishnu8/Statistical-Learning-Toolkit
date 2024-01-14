import numpy as np

class PCA:
    'Principle Component Analysis'

    def __init__(self, data, dim):
        self.data = data
        self.dim = dim
    
    def get_variance_ratio(self):
        data = self.data - np.mean(self.data, axis=0)
        cov = (1/data.shape[0])*np.matmul(np.transpose(data), data)
        cov_eig_values, cov_eig_vecs = np.linalg.eig(cov)
        ratio = (np.sum(cov_eig_values[0:self.dim])/np.sum(cov_eig_values)).real

        return ratio

    def get_features(self):
        #data.shape: n * d

        data = self.data - np.mean(self.data, axis=0)
        cov = (1/data.shape[0])*np.matmul(np.transpose(data), data)
        cov_eig_values, cov_eig_vecs = np.linalg.eig(cov)
        
        principal_eigs = cov_eig_vecs[:,0:self.dim]
        pca_data = np.matmul(np.transpose(principal_eigs), np.transpose(data))
        pca_data = np.transpose(pca_data)
        pca_data = pca_data.real

        #pca_data.shape: n*d'
        return pca_data, principal_eigs