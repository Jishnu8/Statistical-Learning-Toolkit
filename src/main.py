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
# from utils import split_classes
# from utils import plot_2d_data
# from dim_reduction.mda import MDA
# from dim_reduction.pca import PCA
# from param_estimation.mle import MLE
# from param_estimation.bayes_param_estimation import BayesParamEstimation
# from classifers.normal_discriminant import NormalDiscriminant3
from src.pipelines.normal_discriminant import normal_discriminant_classify
from src.pipelines.knn import knn_classify
from src.classifers.pca_approx_method import pca_approximation_method_classify

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    # Normal Discriminant 
    dim_reduction = "MDA"
    projected_dim = 2
    param_estimation = "MLE"
    discriminant_type = "1"
    display_conf = True
    normal_discriminant_classify(x_train, y_train, x_test, y_test, dim_reduction=dim_reduction, 
             projected_dim=projected_dim, discriminant_type=discriminant_type,
             display_conf=display_conf)

    # KNN 
    dim_reduction = "MDA"
    projected_dim = 2
    neighbours=3
    metric_ord = 2
    display_conf = True
    knn_classify(x_train, y_train, x_test, y_test, dim_reduction=dim_reduction, 
                 projected_dim=projected_dim, neighbours=neighbours, metric_ord=metric_ord,
                 display_conf=display_conf)

    #PCA Approx Method
    display_conf = True
    dim = 2
    pca_approximation_method_classify(x_train, y_train, x_test, y_test, dim=dim, display_conf=display_conf)
main()