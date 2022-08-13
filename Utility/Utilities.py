import os
import re
import joblib
import contextlib
from glob import glob
from tqdm import tqdm
from tkinter import Tk, filedialog
from torch import load as torchload
from torch import device as torchdevice
from shapely.geometry import Polygon
import numpy as np
import pickle
import json
from json.decoder import JSONDecodeError
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class SetupOptions:
    def __init__(self):
        self.showPlots = False
        self.doImageRescale = False
        self.imageRescaleHeight = 0
        self.continueTraining = True
        self.modelType = "maskrcnn"
        self.numClasses = 1
        self.folderSuffix = "output"
        self.totalIterations = 10000
        self.iterationCheckpointPeriod = 1000
        self.validationDictPath = ''
        self.trainDictPath = ''
        self.classNameList = ''

        self.imageFilePath = ''
        self.scaleDictPath = ''
        self.modelPath = ''
        self.wireMeasurementsPath = ''
        self.isVerticalSubSection = True
        self.centerFractionToMeasure = 0.5
        self.tiltAngle = 30
        self.scaleBarWidthMicrons = 2
        self.showBoundingBoxPlots = False
        self.plotPolylidar = False
        self.parallelProcessing = True


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


def getJSONFile(basePath, fileName):
    jsonFileName = os.path.join(basePath, fileName)
    if not jsonFileName:
        quit()
    return fileHandling(jsonFileName)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def fileHandling(annotationFileName):
    # with open(annotationFileName, 'rb') as handle:
    #     fileContents = pickle.loads(handle.read())
    with open(annotationFileName, 'r') as fileHandle:
        fileContents = json.load(fileHandle)
    return fileContents


def getFileOrDir(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = None, initialDirOrFile: str = os.getcwd()):
    if os.path.isfile(initialDirOrFile) or os.path.isdir(initialDirOrFile):
        initialDir = os.path.split(initialDirOrFile)[0]
    else:
        initialDir = initialDirOrFile
    root = Tk()
    root.withdraw()
    assert fileOrFolder.lower() == 'file' or fileOrFolder.lower() == 'folder', "Only file or folder is an allowed string choice for fileOrFolder"
    if fileOrFolder.lower() == 'file':
        fileOrFolderList = filedialog.askopenfilename(initialdir=initialDir, title=titleStr, filetypes=[(fileTypes + "file", fileTypes)])
    else:  # Must be folder from assert statement
        fileOrFolderList = filedialog.askdirectory(initialdir=initialDir, title=titleStr)
    if not fileOrFolderList:
        fileOrFolderList = initialDirOrFile
    root.destroy()
    return fileOrFolderList


def getLastIteration(saveDir) -> int:
    """
    Returns:
        int : Number of iterations performed from model in target directory.
    """
    fullSaveDir = os.path.join(os.getcwd(), 'OutputModels', saveDir)
    checkpointFilePath = os.path.join(os.getcwd(), 'OutputModels', fullSaveDir, "last_checkpoint")

    # get file from checkpointFilePath as latestModel
    if os.path.exists(checkpointFilePath):
        with open(checkpointFilePath) as f:
            latestModel = os.path.join(fullSaveDir, f.read().strip())
    elif os.path.exists(os.path.join(fullSaveDir, 'model_final.pth')):
        latestModel = os.path.join(fullSaveDir, 'model_final.pth')
    else:
        fileList = glob("*.pth")
        if fileList:
            latestModel = sorted(fileList, reverse=True)[0]
        else:
            return 0

    latestIteration = torchload(latestModel, map_location=torchdevice("cpu")).get("iteration", -1)
    return latestIteration


def strToFloat(numberString):
    charFreeStr = ''.join(ch for ch in numberString if ch.isdigit() or ch == '.' or ch == ',')
    return float(locale.atof(charFreeStr))


def strToInt(numberString):
    return int(strToFloat(numberString))


def listToCommaString(listValue):
    outString = ""
    for entryNum in range(len(listValue)):
        outString += listValue[entryNum]
        if entryNum < len(listValue) - 1:
            outString += ', '
    return outString


def outputModelFolderConverter(prefix: str, suffix: str):
    return prefix + "Model_" + suffix


def textToBool(text):
    assert text.lower() == 'true' or text.lower() == 'false', "The passed text is not true/false"
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False


def lineSplitter(lineString):
    delimiters = ' ', ', ', ',', '\t', '\n'
    regexPattern = '|'.join(map(re.escape, delimiters))
    splitLineList = re.split(regexPattern, lineString)
    return splitLineList


def checkClassNames(classNamesString, numberClasses):
    splitLine = [entry for entry in lineSplitter(classNamesString) if entry]
    if len(splitLine) != int(numberClasses) or not classNamesString:
        warningColor = (241 / 255, 196 / 255, 15 / 255, 1)
        return False
    else:
        goodColor = (39 / 255, 174 / 255, 96 / 255, 1)
        return True


def bboxToPoly(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def getMaskAndBBDicts(outputs):
    maskDict = {}
    boundingBoxPolyDict = {}
    numInstances = len(outputs['instances'])
    # Loop once to generate dict for checking each instance against all others
    for (mask, boundingBox, instanceNumber) in zip(outputs['instances'].pred_masks, outputs['instances'].pred_boxes, range(numInstances)):
        npBoundingBox = np.asarray(boundingBox.cpu())
        # 0,0 at top left, and box is [left top right bottom] position ie [xmin ymin xmax ymax] (ie XYXY not XYWH)
        boundingBoxPolyDict[instanceNumber] = Polygon(bboxToPoly(npBoundingBox[0], npBoundingBox[1], npBoundingBox[2], npBoundingBox[3]))
        maskDict[instanceNumber] = np.asarray(mask.cpu())
    return maskDict, boundingBoxPolyDict, numInstances


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