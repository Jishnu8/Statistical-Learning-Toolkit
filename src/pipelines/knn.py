import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2
import time
from ..utils import split_classes
from ..utils import plot_2d_data
from ..dim_reduction.mda import MDA
from ..dim_reduction.pca import PCA
from ..param_estimation.mle import MLE
from ..param_estimation.bayes_param_estimation import BayesParamEstimation
from ..classifers.knn import KNN

def knn_classify(x_train, y_train, x_test, y_test, neighbours=3, metric_ord = 2, projected_dim = 9, dim_reduction="MDA", display_conf=False):
    if dim_reduction == "MDA":
        mda = MDA(x_train, y_train, dim=projected_dim) #change to train
        x_train_red, projected_vecs = mda.get_features()
        x_test_red = x_test.dot(projected_vecs)
    elif dim_reduction == "PCA":
        pca = PCA(x_train, dim=projected_dim) #change to train
        x_train_red, projected_vecs = pca.get_features()
        print(pca.get_variance_ratio())
        x_test_red = np.transpose(np.matmul(np.transpose(projected_vecs), np.transpose(x_test) - np.mean(x_test, 0, keepdims=True).T)).real
    elif dim_reduction == None:
        x_train_red = x_train
        x_test_red = x_test
    else:
        raise Exception("Dimensionality reduction type does not exist or is not supported")
    
    knn = KNN(x_train_red, y_train, neighbours, metric_ord)
    knn.classify(x_test_red, y_test, display_conf=display_conf)