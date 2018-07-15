import math

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
