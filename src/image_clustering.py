# Denoise methods
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color, restoration
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
import cv2, os, numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import squareform
import math

ENHANCE_FUNCTION=0
LINKAGE_METHOD = 'ward'
COLOR=1
DEBUG = 0
# Correlation
def corr2(X, Y):
    EXY, EX, EY = np.mean(X*Y), np.mean(X), np.mean(Y)
    VX, VY = np.std(X), np.std(Y)
    return (EXY - EX*EY) / (VX * VY)

def corr(X):
    n = len(X)
    c = []
    for i in range(n-1):
        for j in range(i+1, n):
            c.append([X[i], X[j]])
    return list(map(lambda X: corr2(X[0], X[1]), c))    

# Crop image from center
def cropCenter(im, desire_x, desire_y):
    n_vals = len(im.shape)
    if n_vals == 2:
         h, w = im.shape
    else:
        h, w, _ = im.shape
    if h <= desire_y or w <= desire_x:
        return None
    sW = math.floor((w - desire_x) / 2)
    sH = math.floor((h - desire_y) / 2)
    if n_vals == 2:
        return im[sH:sH+desire_y, sW:sW+desire_x]
    else:
        return im[sH:sH+desire_y, sW:sW+desire_x, :]

# Peak-to-correlation energy for 2 elements
def peak2corr2(X, Y):
    xcorr = cv2.filter2D(X, -1, Y, borderType = cv2.BORDER_WRAP)
    return np.sign(xcorr[0,0]) * xcorr[0,0]**2 / np.mean(xcorr**2)

def peak2corr(X):
    n = len(X)
    p = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            p[i, j] = p[j, i] = peak2corr2(X[i], X[j])
    return p

def zero_mean(data, ztype = 'both'):
    if ztype == 'no':
        return data
    zm = data - np.mean(data, axis=0)
    zm = (zm.T - np.mean(zm, axis=1)).T
    return zm

def wiener_filter(img):
    psf = np.ones((3, 3)) / 9
    return restoration.unsupervised_wiener(img, psf)[0]

def cosine_enhance(noise, alpha = .055):
    # Adjust to [1, 1] interval
    noise = noise / 255
    cpnoise = np.copy(noise)
    # Greater than alpha
    gtalpha = cpnoise > alpha
    ltalpha = np.logical_not(gtalpha)
    # Greater than zero
    gtzero = cpnoise > 0
    ltzero = np.logical_not(gtzero)
    #
    gtmalpha = cpnoise > -alpha
    ltmalpha = np.logical_not(gtmalpha)
    
    cpnoise[gtalpha] = 0
    cpnoise[np.logical_and(gtzero, ltalpha)] = np.cos((np.pi/(2*alpha)) * noise[np.logical_and(gtzero, ltalpha)])
    cpnoise[np.logical_and(ltzero, gtmalpha)] = -np.cos((np.pi/(2*alpha)) * noise[np.logical_and(ltzero, gtmalpha)])
    cpnoise[ltmalpha] = 0
    return cpnoise

def extract_clusters(Z, n):
    CT = cut_tree(Z, n_clusters=n, height=None).flatten()
    #print(CT)
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

def noise_enhance(noise_path, enhance_path, enhance_func):
    if noise_enhance == 0:
        return
    dirlist = os.listdir(noise_path)
    nf = len(dirlist)
    for i in range(nf):
        noise_file = noise_path + '/' + dirlist[i]
        if not os.path.isfile(noise_file):
            continue
        noise = np.load(noise_file)
        noise = enhance_func(noise)
        np.save(enhance_path + '/' + dirlist[i], noise)

def generate_correlation_file(noise_path, results_path):
    filenames = ''
    dirlist = sorted(os.listdir(noise_path))
    nfiles = len(dirlist)
    r = np.ones((nfiles, nfiles))
    # Save element's names
    if DEBUG:
    	print('****CORRELATING****')
    for i in range(nfiles):
        next_file = noise_path + '/' + dirlist[i]
        if DEBUG:
        	print(dirlist[i])
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

'''
    noise_enhance:
        - 0: No enhancing is performed
        - 1: Cosine enhancing (Caldelli et al.)
'''
def cluster_noise(results_path, linkage_method = 'average'):
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
    clusters = minimum_sc(Z, r)
    cluster_file = open(results_path + '/clusters.txt', 'w')
    for i in range(len(clusters)):
        cluster_str = ','.join(list(map(lambda j: elements[j], clusters[i])))
        cluster_file.write(cluster_str + '\n')
    cluster_file.close()
    plot_dendogram(results_path, Z, len(clusters))
    
def plot_dendogram(results_path, Z, n_clusters):
    N = Z.shape[0]
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendogram')
    plt.xlabel('device')
    plt.ylabel('disimilitud')
    dir_vids = os.path.join(results_path, '..', 'vids')
    l = [f for f in sorted(os.listdir(dir_vids)) if os.path.isfile(os.path.join(dir_vids, f))]
    with open(os.path.join(results_path, '..', 'groups.txt')) as f:
        content = f.readlines()
        groups = [x.strip() for x in content]
    labels = [(j+"_"+x.rsplit('_',1)[1]).rsplit('.')[0] for x in l for j in groups if x.startswith(j)]
    dendrogram(
        Z,
        color_threshold=Z[N - n_clusters, 2]+.0001,
        #leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=7.,  # font size for the x axis labels
        orientation='right',
        labels=labels,
    )
    plt.savefig(results_path + '/dendogram.png')
    
def cluster_noise_job(path):
    noise_path = path + '/Noise'
    results_path = path + '/Results'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    # if correlation matrix does not exist
    if ENHANCE_FUNCTION:
        enhance_path = path + '/Enhance'
        if not os.path.isdir(enhance_path):
            os.mkdir(enhance_path)
        if ENHANCE_FUNCTION == 1:
            enhance_func = cosine_enhance
        noise_enhance(noise_path, enhance_path, enhance_func)
        noise_path = enhance_path
    #if not os.path.isfile(results_path + '/corr.npy'):
    generate_correlation_file(noise_path, results_path)
    cluster_noise(results_path, linkage_method = LINKAGE_METHOD)

def denoise_batch(source_path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    noise = []
    dirlist = os.listdir(img_path)
    dirlist = sorted(dirlist)
    if DEBUG:
	    print(dirlist)
    for f in dirlist:
        if f.split('.')[-1].upper() in ['TIF', 'TIFF', 'JPG', 'PNG']:
            if COLOR:
                img = cv2.imread(source_path + '/' + f, 1)
            else:
                img = cv2.imread(source_path + '/' + f, 0)
            img = cropCenter(img, 1024, 1024)
            if img is None:
                print("Image", f, "less than 1024x1024")
                continue
            if DEBUG:
	            print(f)
            if COLOR:
                n0 = img[:,:,0] -  255 * denoise_wavelet(img[:,:,0], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
                n1 = img[:,:,1] -  255 * denoise_wavelet(img[:,:,1], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
                n2 = img[:,:,2] -  255 * denoise_wavelet(img[:,:,2], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
                #cv2 creates image in BGR order
                #n = n0 * 0.2989 + n1 * 0.5870 + n2 * 0.1140
                n = n2 * 0.2989 + n1 * 0.5870 + n0 * 0.1140
            else:
                # Noise
                # Wavelets
                n = img -  255 * denoise_wavelet(img, wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
            # Total Variation
            # n = img -  denoise_tv_chambolle(img.astype(float), weight=10, multichannel=False)
            # Zero-mean
            #n = zero_mean(n)
            # Wiener
            #n = wiener_filter(n)
            # print(noise_path + '/' + f[:-4])
            np.save(dest_path + '/' + f[:-4], n)
            # noise.append(n)


# Whole path
# Denoising
if __name__ == "__main__":
    initial_path = '/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/'
    attempt=None
    with open("cluster.info") as f:
        attempt = f.readlines()[0].strip()
    print("attempt: ", attempt)
    cluster_path = initial_path + attempt
    img_path = cluster_path + '/images'
    noise_path = cluster_path + '/noise'
    print('Denoising')
    denoise_batch(img_path, noise_path)
    print('Clustering')
    cluster_noise_job(cluster_path)
