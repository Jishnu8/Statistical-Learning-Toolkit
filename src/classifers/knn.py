import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, x_train_data, y_train_data, neighbours=3, metric_ord = 2):
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data
        self.neighbours = neighbours
        self.metric_ord = metric_ord #Minkowski metric order

    def __get_distance__(self, x, y):
        if self.metric_ord == np.inf:
            distance = np.max(np.abs(x - y), axis=1)
        else:
            distance = np.sum((np.abs(x - y))**self.metric_ord, axis=1)**(1/self.metric_ord)

        return distance

    def predict(self, X):
        pred = np.array([])
        for i in range(X.shape[0]):
            sample = X[i]
            distances = self.__get_distance__(sample, self.x_train_data)

            smallest_indices = []
            labels = []
            tmp = self.y_train_data
            for j in range(self.neighbours):
                index = np.argmin(distances)
                smallest_indices.append(index)
                distances = np.delete(distances, index)
                labels.append(tmp[index])
                tmp = np.delete(tmp, index)

            pred_index = np.argmax(np.bincount(labels))
            pred = np.append(pred, pred_index)

        return pred

    def classify(self, X, y, display_conf=True):
        pred = self.predict(X)
        acc = accuracy_score(y, pred)
        print("Test accuracy: ", acc)

        if display_conf == True:
            test_conf = confusion_matrix(y, pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=test_conf, display_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            disp.plot()
            plt.title("Test confusion matrix")
            plt.show()

        return acc

