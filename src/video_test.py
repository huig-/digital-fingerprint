import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from utils import cropCenter
import seaborn as sns

CROP_X=CROP_Y=700

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


def correlation_between_frames(vid):
    frames = cv2.VideoCapture(vid)
    while frames.isOpened():
        flag, frame = frames.read()
        if flag:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    frames.release()
    cv2.destroyAllWindows()

def plot_matrix(corr_matrix):
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr_matrix, interpolation='nearest', cmap=cmap)
    ax1.grid(True)
    plt.title('Correlation')
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()
    """
    sns.heatmap(corr_matrix)
    plt.show()


if __name__ == "__main__":
    path = "/users/gago/desktop/universidad/master/tfm/dataset/video/samsung_g6/20160203_161320.mp4"
    correlation_between_frames(path)
    
