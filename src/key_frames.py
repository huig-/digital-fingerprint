from PIL import Image
import numpy as np
import os
from imagehash import phash, average_hash, dhash
from scipy.spatial.distance import squareform

path = "/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Video/bq_aquaris_e5/120311"

fs = os.listdir(path)
hashes = [phash(Image.open(os.path.join(path,f))) for f in fs]
N = len(hashes)
d = np.empty(shape=(N,N))
for i in range(N):
    for j in range(N):
        d[i,j] = hashes[i]-hashes[j]
print(d)
sd = squareform(d)
med = np.median(sd)
print(med)
key_frames = [0]
pos_frames = range(1,N)
last_added = 0


while True:
    candidate_frames = [(d[last_added,j], j) for j in pos_frames if d[last_added,j] > med] 
    pos_frames = [j for j in pos_frames if d[last_added,j] > med]
    if len(candidate_frames) == 0:
        break
    last_added = max(candidate_frames)[1]
    pos_frames.remove(last_added)
    key_frames.append(last_added)
print([1+k for k in key_frames])
