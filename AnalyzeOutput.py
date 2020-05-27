import sys
import os
import pickle
import joblib
import numpy as np
import contextlib
from tqdm import tqdm
import multiprocessing
from matplotlib import cm
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, Button
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.strtree import STRtree
from shapely.prepared import prep
from shapely import affinity
from PIL import Image
from skimage.measure import label, regionprops
from tkinter import Tk, filedialog
from collections import OrderedDict
from polylidarutil import (plot_points, plot_polygons, get_point)
from polylidar import extractPolygons
from uncertainties import unumpy as unp
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from Utility.CropScaleSave import importRawImageAndScale, getNakedNameFromFilePath
from Utility.AnalyzeOutputUI import SetupOptions


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    # noinspection PyProtectedMember
    class TqdmBatchCompletionCallback:
        def __init__(self, _, index, parallel):
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
def centerXPercentofWire(npMaskFunc, setupOptions: SetupOptions):
    # TODO: This is the limiting factor at this point in speed, but it is entirely in the skimage (label_image and allRegionProperties lines) and polylidar (polygonsList line) calls
    assert 0 <= setupOptions.centerFractionToMeasure <= 1, "Percent size of section has to be between 0 and 1"
    assert isinstance(setupOptions.isVerticalSubSection, bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"
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
        if setupOptions.plotPolylidar and not setupOptions.parallelProcessing:
            fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
            # plot points
            plot_points(flippedMaskCoords, ax)
            # plot polygons...doesn't use 2nd argument
            plot_polygons(polygonsList, 0, flippedMaskCoords, ax)
            plt.axis('equal')
            plt.show()

        # Scale the width/height and then skew the squared bounding box by skimage ellipse fitted mask angle about skimage mask centroid
        # This should ensure a measurement line passes through the centroid, at the ellipse angle, and never starts within the mask itself
        # This also means the lengths measured are the real lengths, don't need to do trig later
        centroidCoords = region.centroid
        if setupOptions.isVerticalSubSection:
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, 1, setupOptions.centerFractionToMeasure)
            # Coords are row, col ie (y, x)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, ys=-maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        else:
            # Coords are row, col ie (y, x)
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, setupOptions.centerFractionToMeasure, 1)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, xs=maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        outputSubMaskPoly = maskPolygon.intersection(subBoundingBoxPoly)

        if setupOptions.showBoundingBoxPlots and not setupOptions.parallelProcessing:
            # Blue rectangle is standard bounding box
            # Red rectangle is rotated bounding box from MinimumBoundingBox
            # Multicolored points are either standard (setupOptions.isVerticalSubsection=True) or rotated bounding box (setupOptions.isVerticalSubsection=False)
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111)
            ax.imshow(npMaskFunc)
            r1 = mpatches.Rectangle((xmin, ymax), xmax - xmin, -(ymax - ymin), fill=False, edgecolor="blue", alpha=1, linewidth=1)

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


def isValidLine(strTree, instanceBoxCoords, imageHeight, instanceLine):
    # Check if line is contained in a different bounding box, need to check which instance is in front (below)
    # Don't check for the current instanceNum (key in boundingBoxDict)
    validForInstanceList = []
    bottomLeft, bottomRight, topLeft, topRight = instanceBoxCoords
    lineIntersectingBoxes = strTree.query(instanceLine)
    if lineIntersectingBoxes:
        for bbox in lineIntersectingBoxes:
            # This is faster than doing a index lookup into a list of bounding box polygons then coords
            bottomLeftCheck, bottomRightCheck, topLeftCheck, topRightCheck = getXYFromPolyBox(bbox)
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


def isEdgeInstance(imageRight, imageBottom, instanceBoxCoords, isVerticalSubSection):
    bottomLeft, bottomRight, topLeft, topRight = instanceBoxCoords
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


# @profile
def longestLineAndLengthInPolygon(maskPolygon, lineTest):
    # TODO: This is slow, just the lineTest.intersection line
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
    return xyPoints


# @profile
def getXYFromPolyBox(boundingBoxPoly):
    topXY = []
    bottomXY = []
    boundingBoxXY = boundingBoxPoly.boundary.coords[:-1]
    boundingBoxXYCentroid = boundingBoxPoly.boundary.centroid.coords[0][1]
    assert len(boundingBoxXY) == 4, "The polygon used did not have 4 sides"
    for coords in boundingBoxXY:
        if coords[1] > boundingBoxXYCentroid:
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
def analyzeSingleInstance(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions: SetupOptions):
    if not setupOptions.parallelProcessing:
        print("Working on instance number: ", instanceNumber)

    mask = maskDict[instanceNumber]
    imageWidth = mask.shape[1]
    imageHeight = mask.shape[0]

    measLineList = []
    lineLengthList = []
    lineStd = None
    lineAvg = None

    outputSubMaskPoly, subBoundingBoxPoly, maskAngle = centerXPercentofWire(mask, setupOptions)
    if outputSubMaskPoly is not None:
        strTree = STRtree([poly for i, poly in boundingBoxPolyDict.items() if i != instanceNumber])
        subStrTree = STRtree(strTree.query(boundingBoxPolyDict[instanceNumber]))

        bottomLeft, bottomRight, topLeft, topRight = getXYFromPolyBox(subBoundingBoxPoly)
        instanceBoxCoords = getXYFromPolyBox(boundingBoxPolyDict[instanceNumber])
        if not isEdgeInstance(imageWidth, imageHeight, instanceBoxCoords, setupOptions.isVerticalSubSection):
            if setupOptions.isVerticalSubSection:
                lineStartPoints = getLinePoints(bottomLeft, topLeft)  # Left line
                lineEndPoints = getLinePoints(bottomRight, topRight)  # Right line
            else:
                lineStartPoints = getLinePoints(bottomLeft, bottomRight)  # Bottom line
                lineEndPoints = getLinePoints(topLeft, topRight)  # Top line

            for startPoint, endPoint in zip(lineStartPoints, lineEndPoints):
                instanceLine = LineString([startPoint, endPoint])
                longestLine, lineLength = longestLineAndLengthInPolygon(outputSubMaskPoly, instanceLine)
                if longestLine is not None:
                    if isValidLine(subStrTree, instanceBoxCoords, imageHeight, longestLine):
                        measLineList.append(longestLine)
                        lineLengthList.append(lineLength)
            if len(lineLengthList) > 2:
                # The first and last lines sometimes have issues, remove them
                measLineList = measLineList[1:-1]
                lineLengthList = np.asarray(lineLengthList[1:-1])
                if len(lineLengthList) == 2:
                    lineStd = np.std(lineLengthList, ddof=1)
                else:
                    lineStd = np.std(lineLengthList, ddof=0)
                lineAvg = np.mean(lineLengthList)
            else:
                # else there are no valid lines
                measLineList = []
                lineLengthList = []

    return measLineList, lineLengthList, lineStd, lineAvg, maskAngle


def preparePath(lineObject, instanceColor):
    path = mpath.Path
    x, y = lineObject.xy
    path_data = [(path.MOVETO, [x[0], y[0]]), (path.LINETO, [x[1], y[1]])]
    codes, verts = zip(*path_data)
    outPath = mpath.Path(verts, codes)
    outPatch = mpatches.PathPatch(outPath, color=instanceColor, linewidth=2, alpha=0.5)
    return outPatch


def createPolygonPatchesAndDict(allMeasLineList, isVerticalSubSection):
    numMeasInstances = len(allMeasLineList)
    colorMap = cm.get_cmap('gist_rainbow', numMeasInstances)
    contiguousPolygonsList = []
    patchList = []
    for instanceNumber, lineList in enumerate(allMeasLineList):
        instanceColor = colorMap(instanceNumber)
        contiguousSide1 = []
        contiguousSide2 = []
        for lineNumber, line in enumerate(lineList):
            x, y = line.xy
            if contiguousSide1:
                if isVerticalSubSection:
                    if abs(contiguousSide1[-1][1] - y[0]) < 2 or lineNumber == len(lineList) - 1:
                        contiguousSide1.append((x[0], y[0]))
                        contiguousSide2.append((x[1], y[1]))
                    if abs(contiguousSide1[-1][1] - y[0]) >= 2 or lineNumber == len(lineList) - 1:
                        if len(contiguousSide1) == 1:
                            polygonPerimeter = LineString(contiguousSide1 + contiguousSide2[::-1])
                            outlinePatch = preparePath(polygonPerimeter, instanceColor)
                        else:
                            polygonPerimeter = Polygon(shell=contiguousSide1 + contiguousSide2[::-1])
                            outlinePatch = PolygonPatch(polygonPerimeter, ec='none', fc=instanceColor, fill=True, linewidth=2, alpha=0.5)
                        contiguousPolygonsList.append((instanceNumber, polygonPerimeter))
                        patchList.append(outlinePatch)
                        contiguousSide1 = []
                        contiguousSide2 = []
                else:
                    if abs(contiguousSide1[-1][0] - x[0]) < 2 or lineNumber == len(lineList) - 1:
                        contiguousSide1.append((x[0], y[0]))
                        contiguousSide2.append((x[1], y[1]))
                    if abs(contiguousSide1[-1][0] - x[0]) >= 2 or lineNumber == len(lineList) - 1:
                        if len(contiguousSide1) == 1:
                            polygonPerimeter = LineString(contiguousSide1 + contiguousSide2[::-1])
                            outlinePatch = preparePath(polygonPerimeter, instanceColor)
                        else:
                            polygonPerimeter = Polygon(shell=contiguousSide1 + contiguousSide2[::-1])
                            outlinePatch = PolygonPatch(polygonPerimeter, ec='none', fc=instanceColor, fill=True, linewidth=2, alpha=0.5)
                        contiguousPolygonsList.append((instanceNumber, polygonPerimeter))
                        patchList.append(outlinePatch)
                        contiguousSide1 = []
                        contiguousSide2 = []
            else:
                contiguousSide1.append((x[0], y[0]))
                contiguousSide2.append((x[1], y[1]))
    return contiguousPolygonsList, patchList


class PolygonListManager:
    def __init__(self, contiguousPolygonsList, numInstances, fig, ax):
        self.coords = {}
        self.selectedPolygon = None
        self.numInstances = numInstances
        self.fig = fig
        self.ax = ax
        self.contiguousPolygonsList = contiguousPolygonsList
        self.instancesToMeasSet = set()
        for instanceNumber, _ in self.contiguousPolygonsList:
            self.instancesToMeasSet.add(instanceNumber)

    def RemoveButtonClicked(self, _):
        indicesToDelete = []
        currentInstanceNum = None
        indicesInInstance = []
        deleteCurrentInstance = False
        for regionIndex, (instanceNum, region) in enumerate(self.contiguousPolygonsList):
            if currentInstanceNum != instanceNum:
                if deleteCurrentInstance:
                    indicesToDelete.extend(indicesInInstance)
                currentInstanceNum = instanceNum
                indicesInInstance = []
                deleteCurrentInstance = False
            indicesInInstance.append(regionIndex)
            if region.intersects(self.selectedPolygon):
                deleteCurrentInstance = True
                if instanceNum in self.instancesToMeasSet:
                    self.instancesToMeasSet.remove(instanceNum)
        if deleteCurrentInstance:  # Fix for visual bug of not deleting the last instance
            indicesToDelete.extend(indicesInInstance)
        # Be careful not to mess up indices of list while trying to delete based on index!
        for index in sorted(indicesToDelete, reverse=True):
            del (self.contiguousPolygonsList[index])
            del (self.ax.patches[index])
        self.numInstances = len(self.contiguousPolygonsList)
        del (self.ax.patches[-1])
        rect = RectangleSelector(self.ax, self.RangeSelection, drawtype='box', rectprops=dict(facecolor='red', edgecolor='none', alpha=0.3, fill=True))
        plt.draw()

    def RangeSelection(self, eclick, erelease):
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        self.coords['x'] = [eclick.xdata, erelease.xdata]
        self.coords['y'] = [eclick.ydata, erelease.ydata]
        self.selectedPolygon = Polygon([(eclick.xdata, eclick.ydata), (erelease.xdata, eclick.ydata), (erelease.xdata, erelease.ydata), (eclick.xdata, erelease.ydata)])
        if len(self.ax.patches) > self.numInstances + 1:
            self.ax.patches[-1].remove()
        selection = mpatches.Rectangle((eclick.xdata, eclick.ydata), abs(eclick.xdata - erelease.xdata), abs(eclick.ydata - erelease.ydata), linewidth=1, edgecolor='none', facecolor='red', alpha=0.3, fill=True)
        self.ax.add_patch(selection)
        self.fig.canvas.draw()


def getAnnotationDict(basePath, fileName):
    annotationListFileName = os.path.join(basePath, fileName)
    if not annotationListFileName:
        quit()
    with open(annotationListFileName, 'rb') as handle:
        annotationDict = pickle.loads(handle.read())
    return annotationDict


def getInstances():
    setup_logger()
    basePath = os.getcwd()
    annotationTrainDicts = getAnnotationDict(basePath, "annotations_Train.txt")
    annotationValidateDicts = getAnnotationDict(basePath, "annotations_Validation.txt")
    annotationDicts = [annotationTrainDicts, annotationValidateDicts]

    dirNameSet = set()
    maskTypeSet = set()
    for annotationDictList in annotationDicts:
        for annotationDict in annotationDictList:
            parentDirName = os.path.split(os.path.split(annotationDict['file_name'])[0])[-1]
            if parentDirName not in dirNameSet:
                dirNameSet.add(parentDirName)
            if isinstance(annotationDict['annotations'][0], list):
                fileMaskType = 'polygon'
            elif isinstance(annotationDict['annotations'][0], dict):
                fileMaskType = 'bitmask'
            else:
                fileMaskType = ''
            assert fileMaskType, 'The annotation dict annotations did not match the expected pattern for polygon or bitmask encoding. Check your annotation creation.'
            if fileMaskType not in maskTypeSet:
                maskTypeSet.add(fileMaskType)

    # dirNameSet should return {'Train', 'Validation'}
    assert len(maskTypeSet) == 1, "The number of detected mask types is not 1, check your annotation creation and file choice."
    assert 'Train' in dirNameSet and 'Validation' in dirNameSet, 'You are missing either a Train or Validation directory in your annotations'
    dirnames = ['Train', 'Validation']  # After making sure these are directories as expected, lets force the order to match the annotationDicts order
    nanowireStr = 'VerticalNanowires'
    for d in range(len(dirnames)):
        if nanowireStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
            DatasetCatalog.register(nanowireStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
        MetadataCatalog.get(nanowireStr + "_" + dirnames[d]).set(thing_classes=["VerticalNanowires"])

    DatasetCatalog.get('VerticalNanowires_Train')

    rawImage, scaleBarMicronsPerPixel, setupOptions = importRawImageAndScale()
    if not setupOptions.isVerticalSubSection:
        # Correct for tilt angle, this is equivalent to multiplying the measured length, but is more convenient here
        scaleBarMicronsPerPixel = scaleBarMicronsPerPixel / np.sin(np.deg2rad(setupOptions.tiltAngle))
    npImage = np.array(rawImage)

    if npImage.ndim < 3:
        if npImage.ndim == 2:
            # Assuming black and white image, just copy to all 3 color channels
            npImage = np.repeat(npImage[:, :, np.newaxis], 3, axis=2)
        else:
            print('The imported rawImage is 1 dimensional for some reason, check it out.')
            quit()

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(setupOptions.modelPath)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # (default: 512, balloon test used 128)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (VerticalNanowires)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    predictor = DefaultPredictor(cfg)
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    outputs = predictor(npImage)
    return outputs, npImage, scaleBarMicronsPerPixel * 1000, setupOptions


def main():
    outputs, npImage, scaleBarNMPerPixel, setupOptions = getInstances()
    boundingBoxPolyDict = {}
    maskDict = {}
    numInstances = len(outputs['instances'])
    # Loop once to generate dict for checking each instance against all others
    for (mask, boundingBox, instanceNumber) in zip(outputs['instances'].pred_masks, outputs['instances'].pred_boxes, range(numInstances)):
        npBoundingBox = np.asarray(boundingBox.cpu())
        # 0,0 at top left, and box is [left top right bottom] position ie [xmin ymin xmax ymax] (ie XYXY not XYWH)
        boundingBoxPolyDict[instanceNumber] = Polygon(bboxToPoly(npBoundingBox[0], npBoundingBox[1], npBoundingBox[2], npBoundingBox[3]))
        maskDict[instanceNumber] = np.asarray(mask.cpu())

    if setupOptions.parallelProcessing:
        with joblib.parallel_backend('multiprocessing'):
            with tqdm_joblib(tqdm(desc="Analyzing Instances", total=numInstances)) as progress_bar:
                analysisOutput = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(analyzeSingleInstance)(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions) for instanceNumber in range(numInstances))
    else:
        analysisOutput = []
        for instanceNumber in range(numInstances):
            analysisOutput.append(analyzeSingleInstance(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions))

    allMeasLineList = [entry[0] for entry in analysisOutput if entry[0]]
    contiguousPolygonsList, patchList = createPolygonPatchesAndDict(allMeasLineList, setupOptions.isVerticalSubSection)
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
    plt.imshow(npImage, interpolation='none')
    for patch in patchList:
        ax.add_patch(patch)
    plt.subplots_adjust(bottom=0.15)
    plt.axis('equal')

    polygonListManager = PolygonListManager(contiguousPolygonsList, len(patchList), fig, ax)
    axAddRegion = plt.axes([0.7, 0.02, 0.2, 0.075])
    bRemove = Button(axAddRegion, 'Remove Instance')
    rect = RectangleSelector(ax, polygonListManager.RangeSelection, drawtype='box', rectprops=dict(facecolor='red', edgecolor='none', alpha=0.3, fill=True))
    bRemove.on_clicked(polygonListManager.RemoveButtonClicked)
    plt.show()

    allLineLengthList = [entry[1] for entry in analysisOutput if entry[0]]
    lineStdList = [entry[2] for entry in analysisOutput if entry[0]]
    lineAvgList = [entry[3] for entry in analysisOutput if entry[0]]
    allMeasAnglesList = [entry[4] for entry in analysisOutput if entry[0]]
    finalMeasLineList = []
    finalAllLineLengthList = []
    finalLineStdList = []
    finalLineAvgList = []
    finalAllMeasAnglesList = []
    for instanceNumber in list(polygonListManager.instancesToMeasSet):
        finalMeasLineList.append(allMeasLineList[instanceNumber])
        finalAllLineLengthList.append(allLineLengthList[instanceNumber])
        finalLineStdList.append(lineStdList[instanceNumber])
        finalLineAvgList.append(lineAvgList[instanceNumber])
        finalAllMeasAnglesList.append(allMeasAnglesList[instanceNumber])
    # TODO: The uncertainty calculation isn't quite right, may be doing standard error, need to investigate propagation and inclusion of standard deviation across all the measurements
    # uncertaintyLineArray = unp.uarray(finalLineAvgList, finalLineStdList)
    # averageMeasValue = uncertaintyLineArray.mean()
    averageMeasValue = np.mean(finalLineAvgList)

    print(finalLineAvgList)
    print(finalLineStdList)
    print("Analyzed Image:", getNakedNameFromFilePath(setupOptions.imageFilePath))
    print("Overall Average Size (with std dev): {:.0f} with random standard deviation of {:.0f} nm".format(scaleBarNMPerPixel * averageMeasValue, np.std(finalLineAvgList)))
    print("Number of Measurements: ", len(finalLineAvgList))


if __name__ == "__main__":
    main()