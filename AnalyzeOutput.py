import sys
import os
import pickle
import joblib
import numpy as np
import contextlib
from tqdm import tqdm
import multiprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely import affinity
from PIL import Image
from skimage.measure import label, regionprops
from tkinter import Tk, filedialog
from collections import OrderedDict
from polylidarutil import (plot_points, plot_polygons, get_point)
from polylidar import extractPolygons
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.utils.logger import setup_logger

showPlots = False
showBoundingBoxPlots = False  # Only works if parallel processing is False
plotPolylidar = False  # Only works if parallel processing is False
isVerticalSubSection = False
parallelProcessing = True


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


# @profile
def centerXPercentofWire(npMaskFunc, percentSize, isVerticalSubSection: bool):
    assert 0 <= percentSize <= 1, "Percent size of section has to be between 0 and 1"
    assert isinstance(isVerticalSubSection,
                      bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"
    label_image = label(npMaskFunc, connectivity=1)
    allRegionProperties = regionprops(label_image)
    largeRegionsNums = set()
    regionNum = 0
    for region in allRegionProperties:
        # Ignore small regions or if an instance got split into a major and minor part
        if region.area > 100:
            largeRegionsNums.add(regionNum)
        regionNum += 1
    if len(largeRegionsNums) == 1:
        region = allRegionProperties[list(largeRegionsNums)[0]]
        ymin, xmin, ymax, xmax = region.bbox  # may not need this line

        # maskCords as [row, col] ie [y, x]
        maskCoords = np.array(region.coords)
        flippedMaskCoords = maskCoords.copy()
        flippedMaskCoords[:, 0], flippedMaskCoords[:, 1] = flippedMaskCoords[:, 1], flippedMaskCoords[:, 0].copy()
        maskAngle = np.rad2deg(region.orientation)

        polygonsList = extractPolygons(flippedMaskCoords)
        assert len(polygonsList) == 1, "There was more than 1 polygon extracted from extractPolygons."
        shell_coords = [get_point(pi, flippedMaskCoords) for pi in polygonsList[0].shell]
        maskPolygon = Polygon(shell=shell_coords)
        if plotPolylidar and not parallelProcessing:
            fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
            # plot points
            plot_points(flippedMaskCoords, ax)
            # plot polygons...doesn't use 2nd argument
            plot_polygons(maskPolygon, 0, flippedMaskCoords, ax)
            plt.axis('equal')
            plt.show()

        # Scale the width/height and then skew the squared bounding box by skimage ellipse fitted mask angle about skimage mask centroid
        # This should ensure a measurement line passes through the centroid, at the ellipse angle, and never starts within the mask itself
        # This also means the lengths measured are the real lengths, don't need to do trig later
        centroidCoords = region.centroid
        if isVerticalSubSection:
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, 1, percentSize)
            # Coords are row, col ie (y, x)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, ys=-maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        else:
            # Coords are row, col ie (y, x)
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, percentSize, 1)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, xs=maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        outputSubMaskPoly = maskPolygon.intersection(subBoundingBoxPoly)

        if showBoundingBoxPlots and not parallelProcessing:
            # Blue rectangle is standard bounding box
            # Red rectangle is rotated bounding box from MinimumBoundingBox
            # Multicolored points are either standard (isVerticalSubsection=True) or rotated bounding box (isVerticalSubsection=False)
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111)
            ax.imshow(npMaskFunc)
            r1 = patches.Rectangle((xmin, ymax), xmax-xmin, -(ymax-ymin), fill=False, edgecolor="blue", alpha=1, linewidth=1)

            ax.axis('equal')
            ax.add_patch(r1)

            # 5, since we have 4 points for a rectangle but don't want to have 1st = 4th
            phi = -1 * np.linspace(0, 2*np.pi, 5)
            rgb_cycle = np.vstack((.5 * (1. + np.cos(phi)), .5 * (1. + np.cos(phi + 2 * np.pi / 3)), .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T
            plt.scatter(subBoundingBoxPoly.exterior.coords.xy[0][:-1], subBoundingBoxPoly.exterior.coords.xy[1][:-1], c=rgb_cycle[:4])
            plt.plot(subBoundingBoxPoly.exterior.coords.xy[0], subBoundingBoxPoly.exterior.coords.xy[1])
            plt.autoscale()
            plt.show()

        return outputSubMaskPoly, subBoundingBoxPoly, maskAngle
    # else:
    return None, None, None


def fileHandling(annotationFileName):
    with open(annotationFileName, 'rb') as handle:
        fileContents = pickle.loads(handle.read())
    return fileContents


def bboxToPoly(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


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


# @profile
def isValidLine(boundingBoxPolyDict, imageHeight, instanceNum, instanceLine):
    # Check if line is contained in a different bounding box, need to check which instance is in front (below)
    # Don't check for the current instanceNum (key in boundingBoxDict)
    validForInstanceList = []
    bottomLeft, bottomRight, topLeft, topRight = getXYFromPolyBox(boundingBoxPolyDict[instanceNum])
    for checkNumber, checkBoundingBoxPoly in boundingBoxPolyDict.items():
        if checkNumber != instanceNum:
            # Check if line is contained in a different bounding box, need to check which instance is in front (below)
            lineInvalid = instanceLine.intersects(checkBoundingBoxPoly)

            # May need to do more...something with checking average and deviation from average width of the 2 wires?
            if lineInvalid:
                bottomLeftCheck, bottomRightCheck, topLeftCheck, topRightCheck = getXYFromPolyBox(checkBoundingBoxPoly)

                imageBottom = imageHeight
                instanceBottom = bottomLeft[1]
                checkInstanceBottom = bottomLeftCheck[1]
                if abs(imageBottom - instanceBottom) < 20:
                    # the box of interest is too close to the bottom
                    instanceTop = topLeft[1]
                    checkInstanceTop = topLeftCheck[1]
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


def isEdgeInstance(imageRight, imageBottom, boundingBoxPoly, isVerticalSubSection):
    bottomLeft, bottomRight, topLeft, topRight = getXYFromPolyBox(boundingBoxPoly)
    instanceBottom = bottomLeft[1]
    instanceTop = topLeft[1]
    instanceRight = bottomRight[0]
    instanceLeft = bottomLeft[0]
    if isVerticalSubSection:
        if instanceLeft < 20:
            # too close to left side
            return True
        elif abs(instanceRight - imageRight) < 20:
            # too close to right side
            return True
    else:  # HorizontalSubSection
        if instanceTop < 20:
            # too close to top side
            return True
        elif abs(instanceBottom - imageBottom) < 20:
            # too close to bottom side
            return True
    return False


def longestLineAndLengthInPolygon(maskPolygon, lineTest):
    testSegments = lineTest.intersection(maskPolygon)  # without .boundary, get the lines immediately
    outputLine = None
    LineLength = None
    if isinstance(testSegments, LineString):
        if not testSegments.is_empty:
            # The line is entirely inside the mask polygon
            LineLength = testSegments.length
            outputLine = testSegments
    elif isinstance(testSegments, MultiLineString):
        # The line crosses the boundary of the mask polygon
        LineLength = 0
        for segment in testSegments:
            if segment.length > LineLength:
                LineLength = segment.length
                outputLine = segment
    return outputLine, LineLength


def getLinePoints(startXY, endXY):
    startPoint = Point(startXY)
    endPoint = Point(endXY)
    lineOfInterest = LineString([startPoint, endPoint])

    if sys.version_info < (3, 7):
        xyPoints = OrderedDict()
    else:
        # in Python 3.7 and newer dicts are ordered by default
        xyPoints = {}
    for pos in range(int(np.ceil(lineOfInterest.length)) + 1):
        interpolatedPoint = lineOfInterest.interpolate(pos).coords[0]
        roundedPoint = map(round, interpolatedPoint)
        xyPoints[tuple(roundedPoint)] = None
    # # Can iterated through via:
    # for key in xyPts.keys():
    #     print(key)
    return xyPoints


def getXYFromPolyBox(boundingBoxPoly):
    topXY = []
    bottomXY = []
    boundingBoxXY = boundingBoxPoly.boundary.coords[:-1]
    assert len(boundingBoxXY) == 4, "The polygon used did not have 4 sides"
    for coords in boundingBoxXY:
        if coords[1] > boundingBoxPoly.boundary.centroid.coords[0][1]:
            bottomXY.append(coords)
        else:
            topXY.append(coords)

    if topXY[0][0] > topXY[1][0]:
        topXY.reverse()
    if bottomXY[0][0] > bottomXY[1][0]:
        bottomXY.reverse()
    topLeft = topXY[0]
    topRight = topXY[1]
    bottomLeft = bottomXY[0]
    bottomRight = bottomXY[1]
    return bottomLeft, bottomRight, topLeft, topRight


# @profile
def analyzeSingleInstance(maskDict, boundingBoxPolyDict, instanceNumber, isVerticalSubSection):
    if not parallelProcessing:
        print("Working on instance number: ", instanceNumber)
    mask = maskDict[instanceNumber]
    imageWidth = mask.shape[1]
    imageHeight = mask.shape[0]
    boundingBoxPoly = boundingBoxPolyDict[instanceNumber]
    measLineList = []
    lineLengthList = []
    lineStd = None
    lineAvg = None

    outputSubMaskPoly, subBoundingBoxPoly, maskAngle = centerXPercentofWire(mask, 0.7, isVerticalSubSection)

    if outputSubMaskPoly is not None:
        bottomLeft, bottomRight, topLeft, topRight = getXYFromPolyBox(subBoundingBoxPoly)

        if not isEdgeInstance(imageWidth, imageHeight, boundingBoxPoly, isVerticalSubSection):
            if isVerticalSubSection:
                lineStartPoints = getLinePoints(bottomLeft, topLeft)  # Left line
                lineEndPoints = getLinePoints(bottomRight, topRight)  # Right line
            else:
                lineStartPoints = getLinePoints(bottomLeft, bottomRight)  # Bottom line
                lineEndPoints = getLinePoints(topLeft, topRight)  # Top line

            for startPoint, endPoint in zip(lineStartPoints, lineEndPoints):
                instanceLine = LineString([startPoint, endPoint])
                longestLine, lineLength = longestLineAndLengthInPolygon(outputSubMaskPoly, instanceLine)
                if longestLine is not None:
                    if isValidLine(boundingBoxPolyDict, imageHeight, instanceNumber, longestLine):
                        measLineList.append(longestLine)
                        lineLengthList.append(lineLength)
            if len(lineLengthList) > 2:
                # The first and last lines sometimes have issues, remove them
                measLineList = measLineList[1:-1]
                lineLengthList = np.asarray(lineLengthList[1:-1])
                lineStd = np.std(lineLengthList, ddof=1)
                lineAvg = np.mean(lineLengthList)
            # else there are no valid lines

    return measLineList, lineLengthList, lineStd, lineAvg, maskAngle


# @profile
def main():

    # outputsFileName = getFileOrDirList('file', 'Choose outputs pickle file')
    outputsFileName = '/home/mbraun/Downloads/outputmaskstest'
    outputs = fileHandling(outputsFileName)
    inputFileName = 'tiltedSEM/2020_02_06_MB0232_Reflectometry_002_cropped.jpg'
    if not os.path.isfile(inputFileName):
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

    boundingBoxPolyDict = {}
    maskDict = {}
    numInstances = len(outputs['instances'])
    # Loop once to generate dict for checking each instance against all others
    for (mask, boundingBox, instanceNumber) in zip(outputs['instances'].pred_masks, outputs['instances'].pred_boxes, range(numInstances)):
        npMask = np.asarray(mask.cpu())

        # 0,0 at top left, and box is [left top right bottom] position ie [xmin ymin xmax ymax] (ie XYXY not XYWH)
        npBoundingBox = np.asarray(boundingBox.cpu())
        boundingBoxPoly = Polygon(bboxToPoly(npBoundingBox[0], npBoundingBox[1], npBoundingBox[2], npBoundingBox[3]))

        boundingBoxPolyDict[instanceNumber] = boundingBoxPoly
        maskDict[instanceNumber] = npMask

    if parallelProcessing:
        with joblib.parallel_backend('multiprocessing'):
            with tqdm_joblib(tqdm(desc="Analyzing Instances", total=numInstances)) as progress_bar:
                analysisOutput = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                    joblib.delayed(analyzeSingleInstance)(maskDict, boundingBoxPolyDict, instanceNumber, isVerticalSubSection) for
                    instanceNumber in range(numInstances))
    else:
        analysisOutput = []
        for instanceNumber in range(numInstances):
            analysisOutput.append(analyzeSingleInstance(maskDict, boundingBoxPolyDict, instanceNumber, isVerticalSubSection))

    allMeasLineList = [entry[0] for entry in analysisOutput if entry[0]]
    allLineLengthList = [entry[1] for entry in analysisOutput if entry[0]]
    lineStdList = [entry[2] for entry in analysisOutput if entry[0]]
    lineAvgList = [entry[3] for entry in analysisOutput if entry[0]]
    allMeasAnglesList = [entry[4] for entry in analysisOutput if entry[0]]
    measMask = np.zeros(npImage.shape)[:, :, 0]
    numMeasInstances = len(allMeasLineList)

    colorMap = cm.get_cmap('gist_rainbow', numMeasInstances)

    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    plt.imshow(npImage, interpolation='none')
    contiguousPolygonsDict = {}
    for instanceNumber, lineList in enumerate(allMeasLineList):
        print("instance num:", instanceNumber, "len(linelist):", len(lineList))
        if instanceNumber not in contiguousPolygonsDict:
            contiguousPolygonsDict[instanceNumber] = []
        color = colorMap(instanceNumber)
        contiguousSide1 = []
        contiguousSide2 = []
        for lineNumber, line in enumerate(lineList):
            x, y = line.xy
            if contiguousSide1:
                if isVerticalSubSection:
                    if abs(contiguousSide1[-1][1] - y[0]) < 2 and lineNumber < len(lineList) - 1:
                        contiguousSide1.append((x[0], y[0]))
                        contiguousSide2.append((x[1], y[1]))
                    else:
                        polygonPerimeter = Polygon(shell=contiguousSide1 + contiguousSide2[::-1] + [contiguousSide1[-1]])
                        contiguousPolygonsDict[instanceNumber].append(polygonPerimeter)
                        outlinePatch = PolygonPatch(polygonPerimeter, ec='green', fill=False, linewidth=2)
                        ax.add_patch(outlinePatch)
                        contiguousSide1 = []
                        contiguousSide2 = []
                else:
                    if abs(contiguousSide1[-1][0] - x[0]) < 2 and lineNumber < len(lineList) - 1:
                        contiguousSide1.append((x[0], y[0]))
                        contiguousSide2.append((x[1], y[1]))
                    else:
                        polygonPerimeter = Polygon(shell=contiguousSide1 + contiguousSide2[::-1] + [contiguousSide1[-1]])
                        contiguousPolygonsDict[instanceNumber].append(polygonPerimeter)
                        outlinePatch = PolygonPatch(polygonPerimeter, ec='green', fill=False, linewidth=2)
                        ax.add_patch(outlinePatch)
                        contiguousSide1 = []
                        contiguousSide2 = []
            else:
                contiguousSide1.append((x[0], y[0]))
                contiguousSide2.append((x[1], y[1]))



    # ax.plot(x, y, color=color, linewidth=1)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()