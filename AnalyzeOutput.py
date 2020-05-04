import os
import pickle
import joblib
import contextlib
from tqdm import tqdm
import multiprocessing
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
from os import path
from PIL import Image
from skimage.measure import label, regionprops
from tkinter import Tk, filedialog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
showPlots = False
isVerticalSubSection = True


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


def pointInsidePolygon(x, y, poly):
    # Ray tracing from https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@profile
def centerXPercentofWire(npMaskFunc, percentSize, isVerticalSubSection: bool):
    assert 0 <= percentSize <= 1, "Percent size of section has to be between 0 and 1"
    assert isinstance(isVerticalSubSection,
                      bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"

    label_image = label(npMaskFunc, connectivity=1)
    allRegionProperties = regionprops(label_image)
    subMask = np.zeros(npMaskFunc.shape)
    largeRegionsNums = set()
    regionNum = 0
    for region in allRegionProperties:
        #ignore small regions or if an instance got split into a major and minor part
        if region.area > 100:
            largeRegionsNums.add(regionNum)
        regionNum += 1
    if len(largeRegionsNums) == 1:
        region = allRegionProperties[list(largeRegionsNums)[0]]
        ymin, xmin, ymax, xmax = region.bbox
        if isVerticalSubSection:
            originalbboxHeight = ymax - ymin
            newbboxHeight = originalbboxHeight * percentSize
            ymin = ymin + 0.5 * (originalbboxHeight - newbboxHeight)
            ymax = ymax - 0.5 * (originalbboxHeight - newbboxHeight)
        else:
            originalbboxWidth = xmax - xmin
            newbboxWidth = originalbboxWidth * percentSize
            xmin = xmin + 0.5 * (originalbboxWidth - newbboxWidth)
            xmax = xmax - 0.5 * (originalbboxWidth - newbboxWidth)
        # newBoundingBoxPoly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        newBoundingBoxPoly = bboxToPoly(xmin, ymin, xmax, ymax)
        # maskCords as [row, col] ie [y, x]
        maskCoords = region.coords

        # subMaskCoords as [row, col] ie [y, x]
        subMaskCoords = []
        for row, col in maskCoords:
            if pointInsidePolygon(col, row, newBoundingBoxPoly):
                subMask[row][col] = 1
                subMaskCoords.append((row, col))
        return subMask, subMaskCoords
    # else:
    return None, None



def fileHandling(annotationFileName):
    with open(annotationFileName, 'rb') as handle:
        fileContents = pickle.loads(handle.read())
    return fileContents


def bboxToPoly(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def makeSubMaskCoordsDict(maskCoords, isVerticalSubSection):
    # maskCoords are [row, col]
    assert isinstance(isVerticalSubSection,
                      bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"
    subMaskCoordsDict = {}

    for row, col in maskCoords:
        if isVerticalSubSection:
            if row not in subMaskCoordsDict:
                subMaskCoordsDict[row] = []
            subMaskCoordsDict[row].append(col)
        else:
            if col not in subMaskCoordsDict:
                subMaskCoordsDict[col] = []
            subMaskCoordsDict[col].append(row)
    return subMaskCoordsDict


def getFileOrDirList(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = '*', initialDirOrFile: str = os.getcwd()):
    if os.path.isfile(initialDirOrFile) or os.path.isdir(initialDirOrFile):
        initialDir = os.path.split(initialDirOrFile)[0]
    else:
        initialDir = initialDirOrFile
    root = Tk()
    root.withdraw()
    assert fileOrFolder.lower() == 'file' or fileOrFolder.lower() == 'folder', "Only file or folder is an allowed string choice for fileOrFolder"
    if fileOrFolder.lower() == 'file':
        fileOrFolderList = filedialog.askopenfilename(initialdir=initialDir, title=titleStr,
                                                      filetypes=[(fileTypes + "file", fileTypes)])
    else:  # Must be folder from assert statement
        fileOrFolderList = filedialog.askdirectory(initialdir=initialDir, title=titleStr)
    if not fileOrFolderList:
        fileOrFolderList = initialDirOrFile
    root.destroy()
    return fileOrFolderList


def isValidLine(boundingBoxDict, maskDict, instanceNum, minCoords, maxCoords, isVerticalSubSection):
    # Could use shapeNum (with math for each index call) to eliminate isVerticalSubSection ifs later on
    if isVerticalSubSection:
        shapeNum = 1
    else:
        shapeNum = 0
    # I think this is only working for isVerticalSubSection = True

    # Check min and max coords of each line if it is in any bbox in boundingBoxDict except for the current instanceNum (key in boundingBoxDict)
    validForInstanceList = []
    for checkNumber, checkBoundingBox in boundingBoxDict.items():
        if checkNumber != instanceNum:
            # Check if the start and end of the line hit another bounding box
            # coords are row, col but pointInsidePolygon is x, y
            minCoordInvalid = pointInsidePolygon(minCoords[1], minCoords[0],
                                                 bboxToPoly(checkBoundingBox[0], checkBoundingBox[1],
                                                            checkBoundingBox[2], checkBoundingBox[3]))
            maxCoordInvalid = pointInsidePolygon(maxCoords[1], maxCoords[0],
                                                 bboxToPoly(checkBoundingBox[0], checkBoundingBox[1],
                                                            checkBoundingBox[2], checkBoundingBox[3]))

            # May need to do more...something with checking average and deviation from average width of the 2 wires?
            if minCoordInvalid or maxCoordInvalid:
                imageBottom = maskDict[instanceNum].shape[0]

                instanceTop = boundingBoxDict[instanceNum][1]
                instanceBottom = boundingBoxDict[instanceNum][3]
                checkInstanceTop = checkBoundingBox[1]
                checkInstanceBottom = checkBoundingBox[3]
                if abs(imageBottom - instanceBottom) < 20:
                    # the box of interest is too close to the bottom
                    if instanceTop < checkInstanceTop:
                        # Good assumption that instance of interest is in front, since all wires ~same length and the top is lower than the check instance
                        validForInstanceList.append(True)
                    else:
                        # Maybe need to add more checks if too many lines are being eliminated
                        # intersectingMask = maskDict[checkNumber]
                        validForInstanceList.append(False)
                elif instanceBottom > checkInstanceBottom:
                    # the instance of interest is lower in the image, thus in front due to substrate tilt
                    validForInstanceList.append(True)
                else:
                    validForInstanceList.append(False)
            else:
                validForInstanceList.append(True)
    if all(validForInstanceList):
        return True
    # else:
    return False


def isEdgeInstance(mask, boundingBox, isVerticalSubSection):
    imageRight = mask.shape[1]
    imageBottom = mask.shape[0]
    if isVerticalSubSection:
        if boundingBox[0] < 20:
            # too close to left side
            print('too close to left side')
            return True
        elif abs(boundingBox[2] - imageRight) < 20:
            # too close to right side
            print('too close to right side')
            return True
    else:  # HorizontalSubSection
        if boundingBox[1] < 20:
            # too close to top side
            print('too close to top side')
            return True
        elif abs(boundingBox[3] - imageBottom) < 20:
            # too close to bottom side
            print('too close to bottom side')
            return True
    return False


@profile
def analyzeSingleInstance(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection):
    mask = maskDict[instanceNumber]
    boundingBox = boundingBoxDict[instanceNumber]
    # measCoords will be [row, col]
    measCoordsSet = set()
    if showPlots:
        fig, ax = plt.subplots()
        ax.imshow(mask)
        plt.show(block=False)

    # maskCoords are [row,col] ie [y,x]
    subMask, subMaskCoords = centerXPercentofWire(mask, 0.5, isVerticalSubSection)
    if subMask is not None:
        if showPlots:
            fig2, ax2 = plt.subplots()
            ax2.imshow(subMask)
            plt.show(block=False)

        subMaskCoordsDict = makeSubMaskCoordsDict(subMaskCoords, isVerticalSubSection)

        if not isEdgeInstance(mask, boundingBox, isVerticalSubSection):
            validLineSet = set()
            for line, linePixelsList in subMaskCoordsDict.items():
                if isVerticalSubSection:
                    # Coords as row, col
                    minCoords = (line, min(linePixelsList))
                    maxCoords = (line, max(linePixelsList))
                else:
                    minCoords = (min(linePixelsList), line)
                    maxCoords = (max(linePixelsList), line)

                if isValidLine(boundingBoxDict, maskDict, instanceNumber, minCoords, maxCoords, isVerticalSubSection):
                    validLineSet.add(line)
            for line in validLineSet:
                for value in subMaskCoordsDict[line]:
                    if isVerticalSubSection:
                        measCoordsSet.add((line, value))
                    else:
                        measCoordsSet.add((value, line))

    return measCoordsSet


def main():
    # outputsFileName = getFileOrDirList('file', 'Choose outputs pickle file')
    outputsFileName = '/home/mbraun/Downloads/outputmaskstest'
    outputs = fileHandling(outputsFileName)
    inputFileName = 'tiltedSEM/2020_02_06_MB0232_Reflectometry_002_cropped.jpg'
    if not path.isfile(inputFileName):
        quit()
    rawImage = Image.open(inputFileName)
    npImage = np.array(rawImage)

    if npImage.ndim < 3:
        if npImage.ndim == 2:
            # Assuming black and white image, just copy to all 3 color channels
            npImage = np.repeat(npImage[:, :, np.newaxis], 3, axis=2)
        else:
            print('The imported rawImage is 1 dimensional for some reason, check it out.')
            quit()

    boundingBoxDict = {}
    maskDict = {}
    numInstances = len(outputs['instances'])
    # Loop once to generate dict for checking each instance against all others
    for (mask, boundingBox, instanceNumber) in zip(outputs['instances'].pred_masks, outputs['instances'].pred_boxes, range(numInstances)):
        npMask = np.asarray(mask.cpu())

        # 0,0 at top left, and box is [left top right bottom] position ie [xmin ymin xmax ymax] (ie XYXY not XYWH)
        npBoundingBox = np.asarray(boundingBox.cpu())
        boundingBoxDict[instanceNumber] = npBoundingBox
        maskDict[instanceNumber] = npMask


    # with tqdm_joblib(tqdm(desc="Analyzing Instances", total=numInstances)) as progress_bar:
    #     allMeasCoordsSetList = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
    #         joblib.delayed(analyzeSingleInstance)(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection) for
    #         instanceNumber in range(numInstances))
    allMeasCoordsSetList = []
    for instanceNumber in range(numInstances):
        allMeasCoordsSetList.append(analyzeSingleInstance(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection))

    allMeasCoordsSetList = [entry for entry in allMeasCoordsSetList if entry != set()]
    measMask = np.zeros(npImage.shape)[:, :, 0]
    allMeasCoordsSet = set()
    numMeasInstances = len(allMeasCoordsSetList)
    for coordsSet, instanceNumber in zip(allMeasCoordsSetList, range(numMeasInstances)):
        # print(int(instanceNumber * 255/numMeasInstances))
        for row, col in coordsSet:
            measMask[row][col] = instanceNumber / numMeasInstances
        # allMeasCoordsSet = allMeasCoordsSet.union(coordsSet)


    # https://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    # img_hsv = color.rgb2hsv(img_color)
    # color_mask_hsv = color.rgb2hsv(color_mask)
    #
    # # Replace the hue and saturation of the original image
    # # with that of the color mask
    # img_hsv[..., 0] = color_mask_hsv[..., 0]
    # img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    #
    # img_masked = color.hsv2rgb(img_hsv)

    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.imshow(npImage, cmap=plt.get_cmap('Greys'), interpolation='none')
    plt.imshow(npImage, interpolation='none')

    plt.imshow(measMask, cmap=plt.get_cmap('plasma'), interpolation='none', alpha=0.5)

    # plt.imshow(np.uint8(np.multiply(measMask, 255)), 'jet', interpolation='none', alpha=0.5)
    plt.show()
    # Then put all this in above for loop for (mask, boundingBox, instanceNumber) in zip(outputs['instances'].pred_masks, outputs['instances'].pred_boxes, range(len(outputs['instances']))):


if __name__ == "__main__":
    main()