import os
import imageio
import random
import numpy as np  # (pip install numpy)
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tkinter import Tk, filedialog
from console_progressbar import ProgressBar  # pip install console-progressbar
import multiprocessing
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm
from imgaug.augmentables.batches import UnnormalizedBatch
from collections import deque
from ttictoc import tic, toc
import os
import joblib
import contextlib
import pickle
import multiprocessing
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np  # (pip install numpy)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
from PIL import Image, ImageOps
from tkinter import Tk, filedialog
from skimage.measure import label, regionprops, approximate_polygon, find_contours
from skimage.color import label2rgb
from detectron2.structures import BoxMode
from tqdm import tqdm
import pycocotools
from ttictoc import tic, toc

# Parallelization implemented on ShotNoise, everything else was slower with the number of images used so far.
# Formatting of ShotNoise can be copied easily into the others if wanting to switch back to parallelization
num_cores = multiprocessing.cpu_count()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def getFilesInFolderList(titleString, fileTypeString):
    filesFolder = filedialog.askdirectory(initialdir="/home/mbraun/NewIS", title=titleString)
    if not filesFolder:
        quit()
    (dirpath, dirnames, rawFileNames) = next(os.walk(filesFolder))
    fileNames = []

    for name in rawFileNames:
        if name.endswith(fileTypeString):
            fileNames.append(os.path.join(dirpath, name))
    return fileNames


def convertGrayscaleTo3ChannelFormat(filePath):
    inputImage = imageio.imread(filePath)
    outputImage = np.stack((inputImage,) * 3, axis=-1)
    return outputImage


def getNakedNameFromFilePath(name):
    head, tail = os.path.split(name)
    nakedName, fileExtension = os.path.splitext(tail)
    return nakedName


root = Tk()
root.withdraw()
binaryMaskFileNames = sorted(getFilesInFolderList("Select Binary Mask Image Folder", ".png"))
rawImageFileNames = sorted(getFilesInFolderList("Select Raw Image Folder", ".jpg"))
root.destroy()
# For testing only
# binaryMaskFileNames = ['/home/mbraun/NewIS/MaskedImages/141VerticalWires_Binary.png', '/home/mbraun/NewIS/MaskedImages/MB0239_Center_Binary_SmoothedDilated.png']
# rawImageFileNames = ['/home/mbraun/NewIS/RawImages/2020_03_02_JZL0141_Parallelogram_006_cropped.jpg', '/home/mbraun/NewIS/RawImages/2020_03_05_MB0239_Center_009_cropped.jpg']
if len(binaryMaskFileNames) != len(rawImageFileNames):
    print(
        "There are a different number of files in each of the selected folders. Make sure you have a masked image for each raw image!")
    quit()

for binaryMaskName, rawImageName in zip(binaryMaskFileNames, rawImageFileNames):
    print('raw:', getNakedNameFromFilePath(rawImageName), "binary:", getNakedNameFromFilePath(binaryMaskName))

    rawImage = Image.open(rawImageName)
    binaryImage = Image.open(binaryMaskName)
    assert rawImage.size == binaryImage.size, "Image:" + rawImageName + "and Mask:" + binaryMaskName + "do not have the same image size!"
    rawNPImage = np.array(rawImage)
    maskNPImage = np.array(binaryImage)
    if maskNPImage.ndim > 2:
        if maskNPImage.ndim == 3:
            # Assuming black and white masks, be lazy and only take the first color channel
            maskNPImage = maskNPImage[:, :, 0]
        else:
            print('The imported rawImage is 4 dimensional for some reason, check it out.')
            quit()

    label_image = label(maskNPImage, connectivity=1)
    labeledImageArray = np.array(label_image)
    numberRegions = np.max(labeledImageArray)
    c = matplotlib.cm.get_cmap(name='jet', lut=numberRegions)
    colorList = [c(color)[:3] for color in range(0, numberRegions - 1)]
    colorList.insert(0, (0, 0, 0))
    image_label_overlay = label2rgb(label_image, image=maskNPImage, colors=colorList)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.imshow(rawNPImage, 'gray', interpolation='none')
    plt.imshow(np.uint8(np.multiply(image_label_overlay, 255)), 'jet', interpolation='none', alpha=0.5)
    plt.show()


