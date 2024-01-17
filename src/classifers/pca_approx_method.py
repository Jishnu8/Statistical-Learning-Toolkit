from src.utils import split_classes
from ..dim_reduction.pca import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def get_class_projected_vecs(x_train, y_train):
    '''
    Returns the principal components of the data for each class

    Args:
        x_train: training features
        y_train: training labels
    '''

    class_separated = split_classes(x_train, y_train)
    class_seperated_pca = []
    class_projected_vecs = []
    dim = x_train.shape[1]

    pca = PCA(class_separated[0], dim=dim)
    for i in range(len(class_separated)):
        pca.data = class_separated[i]
        x_train_pca, projected_vecs = pca.get_features()
        class_seperated_pca.append(x_train_pca)
        class_projected_vecs.append(projected_vecs)

    return class_projected_vecs

def pca_approximation_method_classify(x_train, y_train, x_test, y_test, dim=9, display_conf=True):
    '''
    Classifies the test dataset using the pca approximation method 

    Args:
        x_train: training features
        y_train: training labels
        x_test: test features
        y_test: test labels
        dim: the number of principal components used in this method
        display_conf: bool to determine whether the confusion matrix should be displayed.
    '''

    full_class_projected_vecs = get_class_projected_vecs(x_train, y_train)
    class_projected_vecs = []
    for j in range(len(full_class_projected_vecs)):
        class_projected_vecs.append(full_class_projected_vecs[j][:, 0:dim])

    no_of_classes = np.unique(y_train).shape[0]
    class_separated = split_classes(x_train, y_train)
    class_seperated_test_pca = []

    for i in range(no_of_classes):
        x_test_red = np.transpose(np.matmul(np.transpose(class_projected_vecs[i]), np.transpose(x_test).astype(np.float64) - np.mean(class_separated[i], 0, keepdims=True).T))
        class_seperated_test_pca.append(x_test_red)

    class_seperated_test_recon = []
    for i in range(no_of_classes):
        x_test_recon = np.matmul(class_projected_vecs[i], class_seperated_test_pca[i].T) + np.mean(class_separated[i], 0, keepdims=True).T
        x_test_recon = np.real(x_test_recon)
        class_seperated_test_recon.append(x_test_recon.T)

    class_seperated_mse = np.zeros((x_test.shape[0], no_of_classes))
    for i in range(no_of_classes):
        mse = np.mean((class_seperated_test_recon[i].astype(np.float64) - x_test.astype(np.float64))**2, axis=1).real
        class_seperated_mse[:,i] = mse

    predicted_labels = np.argmin(class_seperated_mse, axis=1)
    acc = accuracy_score(predicted_labels, y_test)
    print("Test accuracy: ", acc)

    if display_conf == True:
        test_conf = confusion_matrix(y_test, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=test_conf,
                                      display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

        disp.plot()
        plt.title("Test confusion matrix")
        plt.show()

    return acc