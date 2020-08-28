import os
import re
import joblib
import contextlib
from glob import glob
from tqdm import tqdm
from tkinter import Tk, filedialog
from torch import load as torchload
from torch import device as torchdevice
import pickle
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class SetupOptions:
    def __init__(self):
        self.showPlots = False
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


def getPickleFile(basePath, fileName):
    pickleFileName = os.path.join(basePath, fileName)
    if not pickleFileName:
        quit()
    return fileHandling(pickleFileName)


def fileHandling(annotationFileName):
    with open(annotationFileName, 'rb') as handle:
        fileContents = pickle.loads(handle.read())
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