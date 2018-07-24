import cv2
import os
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from utils import cropCenter
import seaborn as sns
import json

import time

CROP_X=CROP_Y=512

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

def correlation_between_frames(vid, indexes=None):
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
        while frames.isOpened():
            flag, frame = frames.read()
            if flag:
                current_frame = int(frames.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if current_frame in indexes:
                    noise = extract_gprnu_frame(frame).flatten()
                    noise_matrix[cnt , :] = noise
                    cnt += 1
            else:
                break

    frames.release()
    cv2.destroyAllWindows()
    return np.corrcoef(noise_matrix)

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
        i_frames = [x['coded_picture_number'] for x in j_f['frames'] if x['pict_type'] == "I"]
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
    path = "/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Video/samsung_g6/"
    vid = "20160203_161240.mp4"
    path += vid
    ids = find_iframes(path)
    print(ids)
    corr_matrix = correlation_between_frames(path, ids)
    plot_matrix(corr_matrix)
