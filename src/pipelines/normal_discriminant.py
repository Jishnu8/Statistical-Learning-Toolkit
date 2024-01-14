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
from ..classifers.normal_discriminant import NormalDiscriminant1, NormalDiscriminant2, NormalDiscriminant3

def normal_discriminant_classify(x_train, y_train, x_test, y_test, x_val = np.array([]), y_val = np.array([]), dim_reduction="MDA", discriminant_type="1", projected_dim = 9, param_estimation="MLE", display_conf=False):    
    if dim_reduction == "MDA":
        mda = MDA(x_train, y_train, dim=projected_dim) #change to train
        x_train_red, projected_vecs = mda.get_features()
    elif dim_reduction == "PCA":
        pca = PCA(x_train, dim=projected_dim) #change to train
        x_train_red, projected_vecs = pca.get_features()
        print(pca.get_variance_ratio())
    elif dim_reduction == None:
        x_train_red = x_train
    else:
        raise Exception("Dimensionality reduction type does not exist or is not supported")
    
    x_train_red_split = split_classes(x_train_red, y_train)

    if param_estimation == "MLE":
        mus = []
        covs = []
        mle = MLE(x_train_red_split[0], density="gaussian")
        for i in range(len(x_train_red_split)):
            mle.data = x_train_red_split[i]
            mu, cov = mle.get_params()
            mus.append(mu)
            covs.append(cov)
    else:
        raise Exception("Parameter Estimation method does not exist or is not supported")

    train_acc = get_results_normal_discriminant(x_train_red, y_train, mus, covs, dim_reduction, discriminant_type, "Train", display_conf=display_conf)

    #Validation results
    if x_val.any():
        if dim_reduction == "MDA":
            x_val_red = x_val.dot(projected_vecs)
        elif dim_reduction == "PCA":
            x_val_red = np.transpose(np.matmul(np.transpose(projected_vecs),
                                                np.transpose(x_val) - np.mean(x_val, 0, keepdims=True).T)).real
        elif dim_reduction == None:
            x_val_red = x_val

        val_acc = get_results_normal_discriminant(x_val_red, y_val, mus, covs, dim_reduction, discriminant_type, "Validation", display_conf=display_conf)
    else:
        val_acc = 0

    #Testing results
    if dim_reduction == "MDA":
        x_test_red = x_test.dot(projected_vecs)
    elif dim_reduction == "PCA":
        x_test_red = np.transpose(np.matmul(np.transpose(projected_vecs), np.transpose(x_test) - np.mean(x_test, 0, keepdims=True).T)).real
    elif dim_reduction == None:
        x_test_red = x_test

    test_acc = get_results_normal_discriminant(x_test_red, y_test, mus, covs, dim_reduction, discriminant_type, "Test", display_conf=display_conf)
    testing_end = time.time()

    return train_acc, val_acc, test_acc


def get_results_normal_discriminant(x_data, y_data, mus, covs, dim_reduction, discriminant_type, data_type="Train", display_conf=False):
    #assume equal priors

    if (discriminant_type == "1"):
        normal_dis = [NormalDiscriminant1(mus[i]) for i in range(len(mus))]
    elif (discriminant_type == "2"):
        common_cov = covs[0]
        for i in range(1, len(covs)):
            common_cov += covs[i]
        common_cov = common_cov/len(covs)
        normal_dis = [NormalDiscriminant2(mus[i], common_cov) for i in range(len(mus))]

    elif (discriminant_type == "3"):
        normal_dis = [NormalDiscriminant3(mus[i], covs[i], 1/len(mus)) for i in range(len(mus))]    

    pred_data = np.array([])
    for i in range(x_data.shape[0]):
        scores = np.array([])
        for j in range(len(mus)):
            scores = np.append(scores, normal_dis[j].out(x_data[i]))

        pred = np.argmax(np.array(scores))
        pred_data = np.append(pred_data, pred)
    
    acc = accuracy_score(y_data, pred_data)
    if dim_reduction != None:
        print("[" + dim_reduction + " + Gaussian Discriminant function " + discriminant_type + "] " + data_type + " accuracy: ", acc)
    else:
        print("[Gaussian Discriminant function " + discriminant_type + "] " + data_type + " accuracy: ", acc)
    
    if display_conf == True:
        test_conf = confusion_matrix(y_data, pred_data)
        disp = ConfusionMatrixDisplay(confusion_matrix=test_conf, display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        disp.plot()
        plt.title(data_type + " confusion matrix")
        plt.show()

    return acc

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
main()