import os
from sklearn import metrics
from prettytable import PrettyTable
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cluster=None
with open("cluster.info") as f:
    cluster = f.readlines()[0].strip()
print("cluster: ", cluster)

path = "/Users/Gago/Desktop/Universidad/Master/TFM/pruebas"
groups = "groups.txt"
result_path = os.path.join(path, cluster, "Results")

methods = [m for m in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, m))]

cameras = None

with open(os.path.join(path, cluster, groups)) as f:
    cameras = list(enumerate([x.strip() for x in f.readlines()]))

for m in methods:
    print(m)
    results = os.path.join(result_path, m , "clusters.txt")
    corr_path = os.path.join(result_path, "corr.npy")
    elements = os.path.join(result_path, "elements.txt")

    true_labels = []
    pred_labels = []

    with open(results) as f:
        clusters = [x.strip() for x in f.readlines()]
        for i, cluster in enumerate(clusters):
            for image in cluster.split(','):
                true_labels += [s0 for (s0, s1) in cameras if image.startswith(s1)]      
                pred_labels.append(i)

    # Table as in reference papers
    x = PrettyTable()
    x.field_names = ['Model'] + list(range(1, len(clusters) + 1))
    for _, camera in cameras:
        l = []
        for i, cluster in enumerate(clusters):
            l.append(0)
            for image in cluster.split(','):
                if image.startswith(camera): 
                    l[i] += 1
        x.add_row([camera] + l)
    print(x)

    #Table with erros
    y = PrettyTable()
    y.field_names = ['Error', 'ARI', 'MIBS', 'Homogeinity', 'Completeness', 'V-Measure', 'Fowlkes-Mallows']
    y.add_row(['', metrics.adjusted_rand_score(pred_labels, true_labels), metrics.mutual_info_score(true_labels, pred_labels), metrics.homogeneity_score(true_labels, pred_labels), \
        metrics.completeness_score(true_labels, pred_labels),metrics.v_measure_score(true_labels, pred_labels),metrics.fowlkes_mallows_score(true_labels, pred_labels)])
    print(y)
