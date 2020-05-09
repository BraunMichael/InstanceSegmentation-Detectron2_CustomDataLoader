import os
import pickle
import joblib
import contextlib
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely import affinity
import numpy as np
from os import path
from PIL import Image, ImageOps
from skimage.measure import label, regionprops, find_contours
from tkinter import Tk, filedialog
import sys
from collections import OrderedDict
from polylidar import extractPlanesAndPolygons
from polylidarutil import (generate_test_points, plot_points, plot_triangles, get_estimated_lmax,
                           plot_triangle_meshes, get_triangles_from_he, get_plane_triangles, plot_polygons, get_point)

from polylidar import extractPolygons
from MinimumBoundingBox import MinimumBoundingBox
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.utils.logger import setup_logger
showPlots = False
showBoundingBoxPlots = False
plotPolylidar = False
isVerticalSubSection = True
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


def createPolygonFromCoordList(coordList, blankMask):
    # Lets try to find contours and make that a polygon which we need later, and gives us minimum_rotated_rectangle as well
    for coord in coordList:
        blankMask[coord[0]][coord[1]] = 255
    maskCImage = Image.fromarray(np.uint8(blankMask))
    # This fixes the corner issue of diagonally cutting across the mask since edge pixels had no neighboring black pixels
    maskCImage_bordered = ImageOps.expand(maskCImage, border=1)
    contour = find_contours(maskCImage_bordered, 0.5, positive_orientation='low')[0]

    # Flip from (row, col) representation to (x, y)
    # and subtract the padding pixel (not doing that, we didn't buffer)

    # The 2nd line is needed for the correct orientation in the TrainNewData.py file
    # If wanting to showPlots here and get correct orientation, need to change something in the plotting code
    # contour[:, 0], contour[:, 1] = contour[:, 1], sub_mask.size[1] - contour[:, 0].copy()
    contour[:, 0], contour[:, 1] = contour[:, 1], contour[:, 0].copy()

    # Make a polygon and simplify it
    poly = Polygon(contour)
    poly = poly.simplify(1.0, preserve_topology=False)  # should use preserve_topology=True?
    return poly


# @profile
def centerXPercentofWire(npMaskFunc, percentSize, isVerticalSubSection: bool):
    assert 0 <= percentSize <= 1, "Percent size of section has to be between 0 and 1"
    assert isinstance(isVerticalSubSection,
                      bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"
    label_image = label(npMaskFunc, connectivity=1)
    allRegionProperties = regionprops(label_image)
    subMask = np.zeros(npMaskFunc.shape)
    # maskCTest = subMask.copy()
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

        # pip install git+git://github.com/BraunMichael/MinimumBoundingBox.git@master
        # This is much slower
        # shapelyTestRect = MultiPoint(maskCoords).minimum_rotated_rectangle
        # This is a bit faster, but polylidar is faster still
        # maskPolygon = createPolygonFromCoordList(maskCoords, maskCTest)
        # This is actually faster for just the bounding box, but I think slower overall as we need the contour Polygon later
        # outputMinimumBoundingBox = MinimumBoundingBox(maskCoords)

        # Will need to use Polygon subtraction to convert to subMask and final rotated bounding box
        # Extracts planes and polygons, time
        polygonsList = extractPolygons(flippedMaskCoords)
        assert len(polygonsList) == 1, "There was more than 1 polygon extracted from extractPolygons."
        shell_coords = [get_point(pi, flippedMaskCoords) for pi in polygonsList[0].shell]
        maskPolygon = Polygon(shell=shell_coords)
        if plotPolylidar:
            fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
            # plot points
            plot_points(flippedMaskCoords, ax)
            # plot polygons...doesn't use 2nd argument
            plot_polygons(maskPolygon, 0, flippedMaskCoords, ax)
            plt.axis('equal')
            plt.show()

        if isVerticalSubSection:
            # Keep a squared bounding box (envelope) and scale the height
            subBoundingBoxPoly = affinity.scale(maskPolygon.envelope, 1, percentSize)
        else:
            # Scale the width and then skew the squared bounding box by skimage ellipse fitted mask angle about skimage mask centroid
            # This should ensure a measurement line passes through the centroid, at the ellipse angle, and never starts within the mask itself
            # Coords are row, col ie (y, x)
            centroidCoords = region.centroid
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, percentSize, 1)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, xs=maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        outputSubMaskPoly = maskPolygon.intersection(subBoundingBoxPoly)

        # # Do this in isVerticalSubSection is False for length calculations...or maybe also everywhere for less restrictive overlap measures?
        # mbbCenter = outputMinimumBoundingBox['rectangle_center']
        # mbbLength = max(outputMinimumBoundingBox['length_orthogonal'], outputMinimumBoundingBox['length_parallel'])
        # if isVerticalSubSection:
        #     mbbWidth = min(outputMinimumBoundingBox['length_orthogonal'], outputMinimumBoundingBox['length_parallel'])
        # else:
        #     mbbWidth = min(outputMinimumBoundingBox['length_orthogonal'], outputMinimumBoundingBox['length_parallel']) * percentSize
        # lowerLeft = (mbbCenter[1] - mbbWidth / 2, mbbCenter[0] + mbbLength / 2)
        # lowerRight = (mbbCenter[1] + mbbWidth / 2, mbbCenter[0] + mbbLength / 2)
        # upperRight = (mbbCenter[1] + mbbWidth / 2, mbbCenter[0] - mbbLength / 2)
        # upperLeft = (mbbCenter[1] - mbbWidth / 2, mbbCenter[0] - mbbLength / 2)
        # newMBB = Polygon([lowerLeft, lowerRight, upperRight, upperLeft])
        #
        # mbbRotation = outputMinimumBoundingBox['cardinal_angle_deg']
        # rotatedNewMBB = affinity.rotate(newMBB, -mbbRotation)

        if showBoundingBoxPlots:
            # Blue rectangle is standard bounding box
            # Red rectangle is rotated bounding box from MinimumBoundingBox
            # Multicolored points are either standard (isVerticalSubsection=True) or rotated bounding box (isVerticalSubsection=False)
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111)
            ax.imshow(npMaskFunc)
            r1 = patches.Rectangle((xmin, ymax), xmax-xmin, -(ymax-ymin), fill=False, edgecolor="blue", alpha=1, linewidth=1)
            # r2 = patches.Rectangle(lowerLeft, mbbWidth, -mbbLength, fill=False, edgecolor="red", alpha=1, linewidth=1)
            #
            # t2 = mpl.transforms.Affine2D().rotate_deg_around(mbbCenter[1], mbbCenter[0], -mbbRotation) + ax.transData
            # r2.set_transform(t2)
            ax.axis('equal')
            ax.add_patch(r1)
            # ax.add_patch(r2)

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


# @profile
def isValidLine(boundingBoxDict, maskDict, instanceNum, instanceLine):
    # Check if line is contained in a different bounding box, need to check which instance is in front (below)
    # Don't check for the current instanceNum (key in boundingBoxDict)
    validForInstanceList = []
    for checkNumber, checkBoundingBox in boundingBoxDict.items():
        if checkNumber != instanceNum:
            checkPoly = Polygon(bboxToPoly(checkBoundingBox[0], checkBoundingBox[1], checkBoundingBox[2], checkBoundingBox[3]))
            # Check if line is contained in a different bounding box, need to check which instance is in front (below)
            lineInvalid = instanceLine.intersects(checkPoly)

            # May need to do more...something with checking average and deviation from average width of the 2 wires?
            if lineInvalid:
                imageBottom = maskDict[instanceNum].shape[0]
                instanceBottom = boundingBoxDict[instanceNum][3]
                checkInstanceBottom = checkBoundingBox[3]
                if abs(imageBottom - instanceBottom) < 20:
                    instanceTop = boundingBoxDict[instanceNum][1]
                    checkInstanceTop = checkBoundingBox[1]

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


def isEdgeInstance(imageRight, imageBottom, boundingBox, isVerticalSubSection):
    if isVerticalSubSection:
        if boundingBox[0] < 20:
            # too close to left side
            return True
        elif abs(boundingBox[2] - imageRight) < 20:
            # too close to right side
            return True
    else:  # HorizontalSubSection
        if boundingBox[1] < 20:
            # too close to top side
            return True
        elif abs(boundingBox[3] - imageBottom) < 20:
            # too close to bottom side
            return True
    return False


def longestLineLengthInPolygon(maskPolygon, startCoordsRaw, endCoordsRaw):
    startCoord = Point(startCoordsRaw)
    endCoord = Point(endCoordsRaw)
    lineTest = LineString([startCoord, endCoord])
    testSegments = lineTest.intersection(maskPolygon)  # without .boundary, get the lines immediately

    LineLength = None
    if isinstance(testSegments, LineString):
        if not testSegments.is_empty:
            # The line is entirely inside the mask polygon
            LineLength = testSegments.length
    elif isinstance(testSegments, MultiLineString):
        # The line crosses the boundary of the mask polygon
        LineLength = 0
        for segment in testSegments:
            if segment.length > LineLength:
                LineLength = segment.length
    return LineLength


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


# @profile
def analyzeSingleInstance(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection):
    if not parallelProcessing:
        print("Working on instance number: ", instanceNumber)
    mask = maskDict[instanceNumber]
    boundingBox = boundingBoxDict[instanceNumber]
    # measCoords will be [row, col]
    measLineList = []

    # if showPlots:
    #     fig, ax = plt.subplots()
    #     ax.imshow(mask)
    #     plt.show(block=False)

    # maskCoords are [row,col] ie [y,x]
    # subMask, subMaskCoords, maskAngle, rotatedNewMBB = centerXPercentofWire(mask, 0.7, isVerticalSubSection)
    outputSubMaskPoly, subBoundingBoxPoly, maskAngle = centerXPercentofWire(mask, 0.3, isVerticalSubSection)

    if outputSubMaskPoly is not None:
        topXY = []
        bottomXY = []
        for coords in subBoundingBoxPoly.boundary.coords[:-1]:
            if coords[1] > subBoundingBoxPoly.boundary.centroid.coords[0][1]:
                bottomXY.append(coords)
            else:
                topXY.append(coords)

        if topXY[0][0] > topXY[1][0]:
            topXY.reverse()
        if bottomXY[0][0] > bottomXY[1][0]:
            bottomXY.reverse()

        if not isEdgeInstance(mask.shape[1], mask.shape[0], boundingBox, isVerticalSubSection):
            if isVerticalSubSection:
                leftLinePoints = getLinePoints(bottomXY[0], topXY[0])
                rightLinePoints = getLinePoints(bottomXY[1], topXY[1])
                for leftPoint, rightPoint in zip(leftLinePoints, rightLinePoints):
                    instanceLine = LineString([leftPoint, rightPoint])
                    if isValidLine(boundingBoxDict, maskDict, instanceNumber, instanceLine):
                        measLineList.append(instanceLine)

            else:
                # Coords are row, col ie (y,x)
                bottomLinePoints = getLinePoints(bottomXY[0], bottomXY[1])
                topLinePoints = getLinePoints(topXY[0], topXY[1])
                for bottomLinePoint, topLinePoint in zip(bottomLinePoints, topLinePoints):
                    instanceLine = LineString([bottomLinePoint, topLinePoint])
                    if isValidLine(boundingBoxDict, maskDict, instanceNumber, instanceLine):
                        measLineList.append(instanceLine)

    return measLineList, maskAngle


# @profile
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

    if parallelProcessing:
        with tqdm_joblib(tqdm(desc="Analyzing Instances", total=numInstances)) as progress_bar:
            analysisOutput = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(
                joblib.delayed(analyzeSingleInstance)(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection) for
                instanceNumber in range(numInstances))
    else:
        analysisOutput = []
        for instanceNumber in range(numInstances):
            analysisOutput.append(analyzeSingleInstance(maskDict, boundingBoxDict, instanceNumber, isVerticalSubSection))
    allMeasLineList = [entry[0] for entry in analysisOutput if entry[0]]
    allMeasAnglesList = [entry[1] for entry in analysisOutput if entry[0]]

    measMask = np.zeros(npImage.shape)[:, :, 0]
    numMeasInstances = len(allMeasCoordsSetList)
    for coordsSet, instanceNumber in zip(allMeasCoordsSetList, range(numMeasInstances)):
        for row, col in coordsSet:
            measMask[row][col] = (1 + instanceNumber) / numMeasInstances

    # https://stackoverflow.com/questions/17170229/setting-transparency-based-on-pixel-values-in-matplotlib
    measMask = np.ma.masked_where(measMask == 0, measMask)
    plt.subplots(figsize=(15, 12))
    plt.imshow(npImage, interpolation='none')
    plt.imshow(measMask, cmap=plt.get_cmap('plasma'), interpolation='none', alpha=0.5)
    plt.show()

    print('done')


if __name__ == "__main__":
    main()