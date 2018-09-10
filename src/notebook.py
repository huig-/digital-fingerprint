import os
from sklearn import decomposition
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster, inconsistent, maxdists, cophenet
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabaz_score

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def elbow(Z):
    last = Z[-12:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last)+1)
    acceleration = np.diff(last, 2)
    acceleration_rev = acceleration[::-1]
    plt.xticks(idxs)
    plt.plot(idxs, last_rev,'b*-')

def make_dendrogram(Z, labels):
    plt.figure(figsize=(10,6))
    dendrogram(Z, orientation='right', labels = labels, color_threshold=Z[Z.shape[0]-6, 2]+0.001)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("asdjfasdkjfasdfkjasdfk")
    print("jasdjfals;kdjf")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    print(cm)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    title='Normalized confusion matrix')

    plt.show()

if __name__ == "__main__":
    # Compute confusion matrix
    y_test = np.array([1,1,1,1,1,1,1,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,6,6,6,6,6])
    y_pred = np.array([1,1,1,1,1,1,1,2,3,3,2,3,4,4,4,4,4,4,5,5,5,5,6,6,6,6,6])
    class_names = [1,2,3,4,5]
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

    plt.show()

    """
    prueba = 'roc3'
    p = '/Users/Gago/Desktop/Universidad/Master/TFM/pruebas'
    noise_path = os.path.join(p, prueba, 'noise')
    results_path = os.path.join(p, prueba, 'Results')
    c = os.path.join(p, prueba, 'Results', 'corr.npy')
    files = sorted(os.listdir(noise_path))
    s = np.load(c)

    #dis = np.corrcoef(s)
    #dis = np.divide(1+ 1e-10 +s,1 + 1e-10 - s)
    dis = 1-s
    labels = [x for x in sorted(files)]
    possible_labels = ["bq_aquaris_e5", "iphone7", "nexus", "samsung", "xiomi"]
    labels = [possible_labels[i] for i in range(len(possible_labels)) for x in labels if x.startswith(possible_labels[i])]

    Z = linkage(dis, method='ward')
    make_dendrogram(Z, labels)
    """
