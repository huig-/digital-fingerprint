import cv2
from PIL import Image
import imagehash

import os
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from utils import cropCenter
from image_clustering import cluster_noise_job
import seaborn as sns
import json

import time

CROP_X=CROP_Y=710

def extract_noise_frame(frame, color=False):
    cropped_frame = cropCenter(frame, CROP_X, CROP_Y)
    if cropped_frame is None:
        print("Less than", CROP_X, "x", CROP_Y)
        return None
    if color:
        noise_b = cropped_frame[:,:,0] - 255 * denoise_wavelet(cropped_frame[:,:,0], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
        noise_g = cropped_frame[:,:,1] - 255 * denoise_wavelet(cropped_frame[:,:,1], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
        noise_r = cropped_frame[:,:,2] - 255 * denoise_wavelet(cropped_frame[:,:,2], wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
        noise = noise_r * 0.2989 + noise_g * 0.5870 + noise_b * 0.1140
    else:
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        noise = cropped_frame - 255 * denoise_wavelet(cropped_frame, wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
    return noise 

def extract_gprnu_frame(frame):
    cropped_frame = cropCenter(frame, CROP_X, CROP_Y)
    #cropped_frame = cv2.resize(frame, (512,512), interpolation=cv2.INTER_LINEAR)
    if cropped_frame is None:
        print("Less than", CROP_X, "x", CROP_Y)
        return None
    cropped_frame = cropped_frame[:,:,1]
    noise = cropped_frame - 255 * denoise_wavelet(cropped_frame, wavelet='db8', mode='soft', wavelet_levels=4, multichannel=False)
    return noise 

def extract_noise_frames(vid, indexes=None, key_frames=False): #key_frames uses phash
    frames = cv2.VideoCapture(vid)
    noise_matrix = None
    if indexes is None:
        n_frames = int(frames.get(cv2.CAP_PROP_FRAME_COUNT))
        noise_matrix = np.empty(shape=(n_frames, CROP_X*CROP_X))
        while frames.isOpened():
            flag, frame = frames.read()
            if flag:
                noise = extract_noise_frame(frame).flatten()
                noise_matrix[int(frames.get(cv2.CAP_PROP_POS_FRAMES))-1, :] = noise
            else:
                break
    else:
        noise_matrix = np.empty(shape=(len(indexes), CROP_X*CROP_Y))
        cnt = 0
        if key_frames:
            kf_cnt = 0
            hashes = []
        while frames.isOpened():
            flag, frame = frames.read()
            if flag:
                current_frame = int(frames.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_frame in indexes:
                    if key_frames:
                        hashes.append(imagehash.average_hash(Image.fromarray(frame)))   
                    noise = extract_noise_frame(frame).flatten()
                    noise_matrix[cnt , :] = noise
                    cnt += 1
            else:
                break
    """
    if key_frames:
        n = len(indexes)
        print("len", len(hashes))
        similarities = np.empty(shape=(n,n))
        for i in range(n-1):
            for j in range(i+1, n):
                similarities[i,j] = hashes[i] - hashes[j]
        x, y = np.triu_indices(n, 1)
        f_sim = [(i,j, similarities[i,j]) for i,j in list(zip(x,y))]
        ##Get best
        m = np.mean(similarities[x,y])
        real_inds = [0]
        f_sim = [(i,j,k) for (i,j,k) in f_sim if ((i == 0 and k > m) or i!=0)]
        for _ in range(n):
            d = [(val,j) for (i,j,val) in f_sim if i in real_inds and j not in real_inds] + [(val,i) for (i,j,val) in f_sim if i not in real_inds and j in real_inds]
            if len(d) == 0:
                break
            val, j = max(d)
            if val < m:
                break
            real_inds.append(j)
            f_sim = [(i0,j0,k) for (i0,j0,k) in f_sim if (((i0 == j or j0 == j) and k > m) or (i0 != j and j0 !=j))]

        noise_matrix = noise_matrix[real_inds, :]
        print("real_inds", real_inds)
        print("len", len(real_inds))
    """

    frames.release()
    cv2.destroyAllWindows()
    return noise_matrix

def extract_noise_vid(vid, indexes):
    return np.mean(extract_noise_frames(vid, indexes), axis=0)

def correlation_between_captures(path):
    vids = os.listdir(path)
    noise_matrix = np.empty(shape=(len(vids), CROP_X * CROP_X))
    for i, vid in enumerate(vids):
        cap = cv2.VideoCapture(os.path.join(path, vid))
        _, first_frame = cap.read()
        noise = extract_noise_frame(first_frame).flatten()
        noise_matrix[i, :] = noise
        cap.release()
        cv2.destroyAllWindows()
    return np.corrcoef(noise_matrix[:10,:])

def find_iframes(cap_path):
    cap = os.path.basename(cap_path)
    new_file = os.path.join(os.path.dirname(cap_path), os.path.splitext(cap)[0] + ".txt")
    os.system("ffprobe -print_format json -loglevel panic -show_frames -select_streams v {0} > {1}".format(cap_path, new_file))
    with open(new_file) as f:
        j_f = json.load(f)
        i_frames = [n for n, x in enumerate(j_f['frames']) if x['pict_type'] == "I"]
    os.remove(new_file)
    return i_frames

def plot_matrix(corr_matrix):
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(corr_matrix, mask=mask, linewidths=.5)#, cmap="YlGnBu")
        #sns.heatmap(corr_matrix, annot=True, mask=mask, fmt='.1g', annot_kws={"size": 7.5}, linewidths=.5, cmap="YlGnBu")
    plt.show()


if __name__ == "__main__":
    path = "/Users/Gago/Desktop/Universidad/Master/TFM/pruebas/"
    with open("cluster.info") as f:
        attempt = f.readlines()[0].strip()
    print("attempt: ", attempt)
    cluster_path = path + attempt
    video_path = cluster_path + '/vids'
    noise_path = cluster_path + '/noise'
    print('Extracting noise from vids..')
    files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
    for f in files:
        iframes = find_iframes(os.path.join(video_path, f))
        np.save(os.path.join(noise_path, f), extract_noise_vid(os.path.join(video_path, f), iframes)) 
    #print("Clustering")
    #cluster_noise_job(cluster_path)
    """
    path = "/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Video/iphone_8plus/IMG_0545.MOV"
    ids = find_iframes(path)
    print("inds", ids)
    extract_noise_frames(path, ids, True)
    """
