import os
import cv2

path="/Users/Gago/Desktop/Universidad/Master/TFM/Dataset/Dataset-pro/Sony_A57"

onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
onlyfiles = [f for f in onlyfiles if os.path.splitext(f)[1] == ".TIF"]

for f in onlyfiles:
    imx = cv2.imread(os.path.join(path, f))
    filename = os.path.join(path, 'j90', os.path.splitext(f)[0]+".jpeg")
    cv2.imwrite(filename, imx, [int(cv2.IMWRITE_JPEG_QUALITY),90])
