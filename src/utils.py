import numpy as np
import matplotlib.pyplot as plt


def split_classes(data, labels):
    """
    Return data split into seperate classes

    Args:
        data: features
        labels: labels of features

    Returns:
        class_seperated: an array of dimension corresponding to the number of 
        clases. Each entry in the array are the features for that particular class.
    """

    unique_labels = np.unique(labels)
    num_labels = unique_labels.shape[0]
    class_separated = [[] for i in range(num_labels)]

    for sample_id in range(data.shape[0]):
        class_separated[labels[sample_id]].append(data[sample_id])

    for i in range(num_labels):
        class_separated[i] = np.array(class_separated[i]).reshape(len(class_separated[i]), -1)

    return class_separated

def plot_2d_data(data, test_data,  no_of_samples = 100):
    no_of_labels = np.unique(test_data).shape[0]
    colors = np.random.rand(no_of_labels, 3)

    for i in range(no_of_samples):
        plt.scatter(data[i,0], data[i,1], color=colors[test_data[i]])
    plt.show()