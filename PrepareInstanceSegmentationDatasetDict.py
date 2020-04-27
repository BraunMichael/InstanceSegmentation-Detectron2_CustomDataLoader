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

maskType = 'bitmask'  # Options are 'bitmask' or 'polygon'
# If more than 1 type of thing, need a new (and consistent) category_id (in annotate function) for each different type of object
showPlots = False
showSavedMaskAndImage = False
is_crowd = 0  # Likely never relevant for us, used to mark if it is a collection of objects rather than fully separated
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

        # This fixes the corner issue of diagonally cutting across the mask since edge pixels had no neighboring black pixels
        sub_mask_bordered = ImageOps.expand(sub_mask, border=1)
        contours = measure.find_contours(sub_mask_bordered, 0.5, positive_orientation='low')

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
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            # print("Annotation ID:", annotation_id, "poly.geom_type:", poly.geom_type)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

            if showPlots:
                fig, ax = plt.subplots()
                plt.ylim(0, sub_mask.size[1])
                plt.xlim(0, sub_mask.size[0])
                if poly.geom_type is 'Polygon':
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                else:
                    for geom in poly.geoms:
                        xs, ys = geom.exterior.xy
                        ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
                plt.show(block=False)
                plt.pause(1)
                sub_mask_Image = np.array(sub_mask)
                ax.imshow(sub_mask_Image)
                plt.show(block=False)
                plt.pause(1)
                plt.cla()
        if showPlots:
            plt.close()
        # Combine the polygons to calculate the bounding box and areas
        multi_poly = MultiPolygon(polygons)

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


def annotateSingleImage(rawImageName, binaryMaskName, maskType):
    # print('raw:', getNakedNameFromFilePath(rawImageName), "binary:", getNakedNameFromFilePath(binaryMaskName))
    record = {}
    rawImage = Image.open(rawImageName)
    binaryImage = Image.open(binaryMaskName)
    assert rawImage.size == binaryImage.size, "Image:" + rawImageName + "and Mask:" + binaryMaskName + "do not have the same image size!"
    (width, height) = binaryImage.size
    rawNPImage = np.array(rawImage)
    image = np.array(binaryImage)
    if image.ndim > 2:
        if image.ndim == 3:
            # Assuming black and white masks, be lazy and only take the first color channel
            image = image[:, :, 0]
        else:
            print('The imported rawImage is 4 dimensional for some reason, check it out.')
            quit()

    record["file_name"] = rawImageName
    record["image_id"] = rawImageName
    record["height"] = height
    record["width"] = width
    # label image regions
    label_image = label(image, connectivity=1)
    if showPlots or showSavedMaskAndImage:
        labeledImageArray = np.array(label_image)
        numberRegions = np.max(labeledImageArray)
        c = matplotlib.cm.get_cmap(name='jet', lut=numberRegions)
        colorList = [c(color)[:3] for color in range(0, numberRegions - 1)]
        colorList.insert(0, (0, 0, 0))
        image_label_overlay = label2rgb(label_image, image=image, colors=colorList)
    if showPlots:
        rgbImage = Image.fromarray(np.uint8(np.multiply(image_label_overlay, 255)))
        rgbImage.show()
        fig, ax = plt.subplots(figsize=(10, 8))

    allRegionProperties = regionprops(label_image)

    objectsList = []
    regionNumber = 1
    for region in allRegionProperties:
        # take regions with large enough areas
        if region.area > 100:
            category_id = 0  # If more than 1 type of thing, need a new (and consistent) category_id for each different type of object
            maskCoords = region.coords
            subMask = np.zeros(image.shape)
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
    binaryFilesFolder = filedialog.askdirectory(title="Select Binary Mask Image Folder")
    rawFilesFolder = filedialog.askdirectory(title="Select Raw Image Folder")
    root.destroy()
    if not binaryFilesFolder or not rawFilesFolder:
        quit()

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
            if name.endswith(('.png')):
                binaryImageNames.append(os.path.join(binaryDirpath, name))

        rawImageNames = sorted(rawImageNames)
        binaryImageNames = sorted(binaryImageNames)
        assert len(binaryImageNames) == len(rawImageNames), "There are a different number of files in each of the selected folders. Make sure you have a masked image for each raw image! Make sure you have png masks!"

        totalImages = len(binaryImageNames)
        with tqdm_joblib(tqdm(desc="Annotating" + dirName + "Images", total=totalImages)) as progress_bar:
            allAnnotations = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(annotateSingleImage)(rawImageName, binaryImageName, maskType) for (rawImageName, binaryImageName) in zip(rawImageNames, binaryImageNames))

        # allAnnotationsDict = {key: value for i in allAnnotations for key, value in i.items()}
        annotationDictFileName = 'new_annotations_dict_bitmask_' + dirName + '.txt'
        with open(annotationDictFileName, 'wb') as handle:
            pickle.dump(allAnnotations, handle)


if __name__ == "__main__":
    main()