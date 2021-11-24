import os
import joblib
import contextlib
import pickle
import multiprocessing
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np  # (pip install numpy)
from statistics import mode
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
from PIL import Image, ImageOps
from tkinter import Tk, filedialog
from skimage.measure import label, regionprops, approximate_polygon, find_contours
from skimage.color import label2rgb
from detectron2.structures import BoxMode
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import pycocotools
from ttictoc import tic, toc

maskType = 'polygon'  # Options are 'bitmask' or 'polygon'
# If more than 1 type of thing, need a new (and consistent) category_id (in annotate function) for each different type of object
showPlots = False
showSavedMaskAndImage = False
is_crowd = 0  # Likely never relevant for us, used to mark if it is a collection of objects rather than fully separated
num_cores = multiprocessing.cpu_count()
parallelProcessing = True
assert maskType.lower() == 'bitmask' or maskType.lower() == 'polygon', "The valid maskType options are 'bitmask' and 'polygon'"


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


# noinspection PyShadowingNames
def create_sub_mask_annotation(sub_mask, region, category_id, annotation_id, is_crowd, maskType):
    sub_mask = sub_mask.astype('uint8')
    minr, minc, maxr, maxc = region.bbox
    if maskType.lower() == 'bitmask':
        annotationDict = {
            'segmentation': pycocotools.mask.encode(np.array(sub_mask, order="F")),
            'iscrowd': is_crowd,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': [minc, minr, maxc, maxr],
            'bbox_mode': BoxMode.XYXY_ABS,
            'area': region.area
        }
    else:
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        subMaskImage = Image.fromarray(np.uint8(np.multiply(sub_mask, 255)))
        # This fixes the corner issue of diagonally cutting across the mask since edge pixels had no neighboring black pixels
        sub_mask_bordered = ImageOps.expand(subMaskImage, border=1)
        contours = measure.find_contours(np.array(sub_mask_bordered), 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel (not doing that, we didn't buffer)

            # The 2nd line is needed for the correct orientation in the TrainNewData.py file
            # If wanting to showPlots here and get correct orientation, need to change something in the plotting code
            # contour[:, 0], contour[:, 1] = contour[:, 1], sub_mask.size[1] - contour[:, 0].copy()
            contour[:, 0], contour[:, 1] = contour[:, 1], contour[:, 0].copy()

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)  # should use preserve_topology=True?
            polygons.append(poly)
            # print("Annotation ID:", annotation_id, "poly.geom_type:", poly.geom_type)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

            if showPlots:
                fig, ax = plt.subplots()
                plt.ylim(0, subMaskImage.size[1])
                plt.xlim(0, subMaskImage.size[0])
                if poly.geom_type == 'Polygon':
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                else:
                    for geom in poly.geoms:
                        xs, ys = geom.exterior.xy
                        ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                plt.show(block=False)
                plt.pause(1)
                npSubMaskImage = np.array(subMaskImage)
                ax.imshow(npSubMaskImage)
                plt.show(block=False)
                plt.pause(1)
                plt.cla()
        if showPlots:
            plt.close()
        # Combine the polygons to calculate the bounding box and areas
        multi_poly = MultiPolygon(polygons)
        assert len(segmentations) == 1, "Length of segmentations must be 1"  # Otherwise MaskRCNN breaks, this should be taken care of with binary_fill_holes

        annotationDict = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': multi_poly.bounds,
            'bbox_mode': BoxMode.XYXY_ABS,
            'area': multi_poly.area
        }
    return annotationDict


def annotateSingleImage(rawImageName, binaryMaskName, maskType, parentFolder):
    # print('raw:', getNakedNameFromFilePath(rawImageName), "binary:", getNakedNameFromFilePath(binaryMaskName))
    record = {}
    rawImage = Image.open(rawImageName)
    binaryImage = Image.open(binaryMaskName)
    assert rawImage.size == binaryImage.size, "Image:" + rawImageName + "and Mask:" + binaryMaskName + "do not have the same image size!"
    (width, height) = binaryImage.size
    rawNPImage = np.array(rawImage)
    binaryNPImageOriginal = np.array(binaryImage)
    if binaryNPImageOriginal.ndim > 2:
        if binaryNPImageOriginal.ndim == 3:
            # Assuming black and white masks, be lazy and only take the first color channel
            binaryNPImageOriginal = binaryNPImageOriginal[:, :, 0]
        else:
            print('The imported rawImage is 4 dimensional for some reason, check it out.')
            quit()
    binaryNPImage = binary_fill_holes(binaryNPImageOriginal)  # Fill holes in image, MaskRCNN polygon maskType breaks with holes
    record["file_name"] = os.path.relpath(rawImageName, parentFolder)
    record["image_id"] = os.path.relpath(rawImageName, parentFolder)
    record["height"] = height
    record["width"] = width
    # label image regions
    label_image = label(binaryNPImage, connectivity=1)
    if showPlots or showSavedMaskAndImage:
        labeledImageArray = np.array(label_image)
        numberRegions = np.max(labeledImageArray)
        c = matplotlib.cm.get_cmap(name='jet', lut=numberRegions)
        colorList = [c(color)[:3] for color in range(0, numberRegions - 1)]
        colorList.insert(0, (0, 0, 0))
        image_label_overlay = label2rgb(label_image, image=binaryNPImage, colors=colorList)
    if showPlots:
        rgbImage = Image.fromarray(np.uint8(np.multiply(image_label_overlay, 255)))
        rgbImage.show()
        fig, ax = plt.subplots(figsize=(10, 8))

    allRegionProperties = regionprops(label_image)
    hist, bins = np.histogram(binaryNPImageOriginal.ravel(), np.max(binaryNPImageOriginal) + 1, [0, np.max(binaryNPImageOriginal) + 1])
    levelDict = {}
    for ind, value in enumerate(np.where(hist > 0)[0]):
        levelDict[value] = ind - 1  # Assumes a black background (so the 0 index is not a class)
    objectsList = []
    regionNumber = 1
    for region in allRegionProperties:
        # take regions with large enough areas
        if region.area > 100:
            regionGreyLevel = mode([binaryNPImageOriginal[coord[0],coord[1]] for coord in region.coords])
            category_id = levelDict[regionGreyLevel]
            assert category_id >= 0, "Code assumes a black background that is not a designated class"
            maskCoords = region.coords

            subMask = np.zeros(binaryNPImage.shape)
            for pixelXY in maskCoords:
                subMask[pixelXY[0]][pixelXY[1]] = 1
            annotationDict = create_sub_mask_annotation(subMask, region, category_id, regionNumber, 0, maskType)
            objectsList.append(annotationDict)

            if showPlots:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red',
                                          linewidth=2)
                # noinspection PyUnboundLocalVariable
                ax.imshow(subMask)
                ax.add_patch(rect)
                plt.show(block=False)
                plt.pause(0.5)
                plt.cla()
        else:
            # print("Region", annotation_id, " area is:", region.area, ". Not updating annotation_id")
            pass
        regionNumber += 1
    if showPlots:
        plt.close()
    if showSavedMaskAndImage:
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(rawNPImage, 'gray', interpolation='none')
        plt.imshow(np.uint8(np.multiply(image_label_overlay, 255)), 'jet', interpolation='none', alpha = 0.5)
        plt.show()

    record["annotations"] = objectsList
    return record


def getNakedNameFromFilePath(name):
    head, tail = os.path.split(name)
    nakedName, fileExtension = os.path.splitext(tail)
    return nakedName


def main():
    root = Tk()
    root.withdraw()
    # Folder structure should be a mask folder, inside a train and validation folder, inside each of those are the actual images
    # A duplicate folder structure should exist for the raw images
    binaryFilesFolder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Binary Mask Image Folder")
    if not binaryFilesFolder:
        root.destroy()
        print("No binaryfilesfolder")
        quit()
    rawFilesFolder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select Raw Image Folder")
    if not rawFilesFolder:
        root.destroy()
        print("No rawfilesfolder")
        quit()
    root.destroy()
    parentFolder, _ = os.path.split(binaryFilesFolder)
    altParentFolder, _ = os.path.split(rawFilesFolder)
    assert parentFolder == altParentFolder, "The parent folder of the chose binary and raw folders do not match, check your folder structure"
    (_, binaryDirnames, _) = next(os.walk(binaryFilesFolder))
    (_, rawDirnames, _) = next(os.walk(rawFilesFolder))

    binaryDirnames = sorted(binaryDirnames)
    rawDirnames = sorted(rawDirnames)
    assert binaryDirnames == rawDirnames, "Make sure you have the same number of directories, and with identical names, in both folders!"

    for dirName in binaryDirnames:
        (binaryDirpath, _, binaryRawFileNames) = next(os.walk(os.path.join(binaryFilesFolder, dirName)))
        (rawDirpath, _, rawFileNames) = next(os.walk(os.path.join(rawFilesFolder, dirName)))

        rawImageNames = []
        binaryImageNames = []
        for name in rawFileNames:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                rawImageNames.append(os.path.join(rawDirpath, name))
        for name in binaryRawFileNames:
            if name.endswith('.png'):
                binaryImageNames.append(os.path.join(binaryDirpath, name))

        rawImageNames = sorted(rawImageNames)
        binaryImageNames = sorted(binaryImageNames)
        assert len(binaryImageNames) == len(rawImageNames), "There are a different number of files in each of the selected folders. Make sure you have a masked image for each raw image! Make sure you have png masks!"

        totalImages = len(binaryImageNames)

        if parallelProcessing:
            with tqdm_joblib(tqdm(desc="Annotating" + dirName + "Images", total=totalImages)) as progress_bar:
                allAnnotations = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(annotateSingleImage)(rawImageName, binaryImageName, maskType, parentFolder) for (rawImageName, binaryImageName) in zip(rawImageNames, binaryImageNames))
        else:
            allAnnotations = []
            for (rawImageName, binaryImageName) in zip(rawImageNames, binaryImageNames):
                allAnnotations.append(annotateSingleImage(rawImageName, binaryImageName, maskType, parentFolder))

        annotationDictFileName = 'annotations_16NoSnNoMerged_' + dirName + '.txt'
        with open(annotationDictFileName, 'wb') as handle:
            pickle.dump(allAnnotations, handle)


if __name__ == "__main__":
    main()