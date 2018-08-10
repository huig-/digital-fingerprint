import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

def generate_correlation_file(noise_path, results_path):
    filenames = ''
    dirlist = sorted(os.listdir(noise_path))
    nfiles = len(dirlist)
    r = np.ones((nfiles, nfiles))
    # Save element's names
    for i in range(nfiles):
        next_file = noise_path + '/' + dirlist[i]
        np1 = np.load(next_file).flatten()
        filenames = filenames + ',' + dirlist[i].split('.')[0]
        for j in range(i + 1, nfiles):
            next_file = noise_path + '/' + dirlist[j]
            if not os.path.isfile(next_file):
                continue
            np2 = np.load(next_file).flatten()
            r[i, j] = r[j, i] = np.corrcoef(np1, np2)[0, 1]
    np.save(results_path + '/corr', r)
    filenames = filenames[1:]
    f = open(results_path + '/elements.txt', 'w')
    f.write(filenames)
    f.close()

def generate_mahalanobis(noise_path, results_path):
    dirlist = sorted(os.listdir(noise_path))
    nfiles = len(dirlist)
    r = np.empty(shape=(nfiles, 700*700))
    for i in range(nfiles):
        noise = np.load(os.path.join(noise_path, dirlist[i]))
        r[i,:] = noise
    np.save(os.path.join(results_path, 'noises'), r)

    n = r.shape[0]
    dd = np.empty((n,n))
    cov = np.linalg.inv(np.cov(r, rowvar=False))
    for i in range(n):
        for j in range(i+1, n):
            dd[i,j] = mahalanobis(r[i,:], r[j,:], cov[i,j])
    print(dd)

def extract_clusters(Z, n):
    CT = cut_tree(Z, n_clusters=n, height=None).flatten()
    clusters = []
    for i in range(n):
        clusters.append(np.where(CT == i)[0])
    return clusters

def silhouette_coefficient(clusters, r):
    N =  sum(map(len, clusters))
    idx = np.arange(N)
    a = np.zeros(N)
    b = np.zeros(N)
    for C in np.array(list(map(np.sort, clusters))):
        D = list(filter(lambda x: x not in C, idx))
        m = len(C)
        n = len(D)
        for i in range(m):
            # Cohesion
            for j in range(i):
                a[C[i]] += r[C[i], C[j]]
            for j in range(i+1, m):
                a[C[i]] += r[C[i], C[j]]
            a[C[i]] /= max(1, m)
            # Separation
            for j in range(n):
                b[C[i]] += r[C[i], D[j]]
            b[C[i]] /= max(1, m)
    s = b-a
    return sum(s) / N

def minimum_sc(Z, r):
    N = Z.shape[0]
    sc = np.inf
    min_clusters = None
    for i in range(1, N+1):
        # Gets the clusters defined by the i-th tree level in hthe hierarchy
        clusters = extract_clusters(Z, i)
        sc_i = silhouette_coefficient(clusters, r)
        if (sc_i < sc):
            sc = sc_i
            min_clusters = clusters
    return min_clusters

def cluster_noise(results_path, linkage_method):
    # Read element names
    f = open(results_path + '/elements.txt', 'r')
    elements = f.read().split(',')
    f.close()
    r = np.load(results_path + '/corr.npy')
    disimilitude = 1 - r
    disimilitude = squareform(disimilitude)
    Z = linkage(disimilitude, method=linkage_method)
    Z[:, 2] = Z[:, 2]-Z[:, 2].min()
    # Clusters are separated according to Caldelli's paper. This should be changed to allow more mmethods

    #clusters = minimum_sc(Z, r)

    sss = -10
    sss_i = 0
    for i in range(2, Z.shape[0]+1):
        nodes = fcluster(Z, i, criterion="maxclust")
        si = silhouette_score(1-r, nodes, metric='euclidean')
        if (si > sss):
            sss = si
            sss_i = i
    clusters = extract_clusters(Z, sss_i)

    cluster_file = open(results_path + '/' + linkage_method + '/clusters.txt', 'w')
    for i in range(len(clusters)):
        cluster_str = ','.join(list(map(lambda j: elements[j], clusters[i])))
        cluster_file.write(cluster_str + '\n')
    cluster_file.close()
    plot_dendogram(os.path.join(results_path, linkage_method), Z, len(clusters))
    
def plot_dendogram(path, Z, n_clusters):
    N = Z.shape[0]
    plt.figure(figsize=(12, 6))
    plt.title('Hierarchical Clustering Dendogram')
    plt.ylabel('device')
    plt.xlabel('disimilitud')
    dir_noise = os.path.join(path, '..', '..', 'noise')
    l = [f for f in sorted(os.listdir(dir_noise)) if os.path.isfile(os.path.join(dir_noise, f))]
    with open(os.path.join(path, '..', '..', 'groups.txt')) as f:
        content = f.readlines()
        groups = [x.strip() for x in content]
    labels = [(j+"_"+x.rsplit('_',1)[1]).rsplit('.')[0] for x in l for j in groups if x.startswith(j)]
    """
    cmap = cm.rainbow(np.linspace(1,len(groups)+1,len(groups)+1))
    leaf_colors = {}
    for l in labels:
        i = [i for i,j in enumerate(groups) if l.startswith(j)][0]
        leaf_colors[l] = cmap[i-1]
    print(leaf_colors) 
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
        c1, c2 = (link_cols[x] if x > len(Z) else leaf_colors[labels[x]] for x in i12)
        print(c1,c2)
        link_cols[i+1+len(Z)] = c1 if (c1 == c2).all() else cmap[len(groups)]
    """
    dendrogram(
        Z,
        color_threshold=Z[N - n_clusters, 2]+.0001,
        #leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        orientation='right',
        labels=labels,
        #link_color_func=lambda x: link_cols[x]
    )
    plt.savefig(path + '/dendogram.png')

if __name__ == "__main__":
    initial_path = '/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/'
    attempt = None
    with open("cluster.info") as f:
        attempt = f.readlines()[0].strip()
    print("attempt", attempt)
    noise_path = os.path.join(initial_path, attempt, 'noise')
    results_path = os.path.join(initial_path, attempt, 'Results')
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    #generate_mahalanobis(noise_path, results_path)
    generate_correlation_file(noise_path, results_path)
    #methods = ['single', 'complete', 'average', 'weighted', 'ward']
    methods = ['ward']
    for m in methods:
        print("method", m)
        if not os.path.isdir(os.path.join(results_path,m)):
            os.makedirs(os.path.join(results_path, m))
        cluster_noise(results_path, m)
