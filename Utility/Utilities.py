import os
import joblib
import contextlib
from tqdm import tqdm
from tkinter import Tk, filedialog
import pickle


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



