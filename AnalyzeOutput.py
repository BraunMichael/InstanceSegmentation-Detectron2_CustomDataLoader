import json
import sys
import os
import joblib
import shutil
import pyp3rclip
import numpy as np
from tqdm import tqdm
import multiprocessing
from matplotlib import cm
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, Button
from mpl_toolkits.axes_grid1 import ImageGrid
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, MultiLineString, Polygon

from PIL import Image
from uncertainties import unumpy as unp
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from Utility.CropScaleSave import importRawImageAndScale, getNakedNameFromFilePath
# from Utility.AnalyzeOutputUI import SetupOptions
from Utility.AnalyzeTiltInstance import analyzeSingleTiltInstance
from Utility.AnalyzeTopDownInstance import analyzeTopDownInstance
from Utility.ImageGridding import splitSingleImage, create_collage
from Utility.Utilities import *


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
                            outlinePatch = PolygonPatch(polygonPerimeter, ec=instanceColor, fc=instanceColor, fill=True, lw=5, alpha=0.5)
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
                            outlinePatch = PolygonPatch(polygonPerimeter, ec=instanceColor, fc=instanceColor, fill=True, lw=5, alpha=0.5)
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





def getInstances():
    setup_logger()
    basePath = os.getcwd()

    # TODO: some or all of this may not be necessary
    annotationTrainDicts = getJSONFile(basePath, "annotations_Train.json")
    annotationValidateDicts = getJSONFile(basePath, "annotations_Validation.json")
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
    assert len(maskTypeSet) == 1, "The number of detected mask types is not 1, check your annotation creation and file choice."
    # dirNameSet should return {'Train', 'Validation'}
    assert 'Train' in dirNameSet and 'Validation' in dirNameSet, 'You are missing either a Train or Validation directory in your annotations'
    dirnames = ['Train', 'Validation']  # After making sure these are directories as expected, lets force the order to match the annotationDicts order

    rawImage, scaleBarMicronsPerPixel, setupOptions = importRawImageAndScale()
    if not setupOptions.isVerticalSubSection and not setupOptions.tiltAngle == 0:
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

    if setupOptions.tiltAngle == 0:
        nanowireStr = 'TopDownNanowires'
        for d in range(len(dirnames)):
            if nanowireStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
                DatasetCatalog.register(nanowireStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
            MetadataCatalog.get(nanowireStr + "_" + dirnames[d]).set(thing_classes=setupOptions.classNameList)

    else:
        nanowireStr = 'VerticalNanowires'
        for d in range(len(dirnames)):
            if nanowireStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
                DatasetCatalog.register(nanowireStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
            MetadataCatalog.get(nanowireStr + "_" + dirnames[d]).set(thing_classes=["VerticalNanowires"])

    DatasetCatalog.get(nanowireStr+'_Train')
    nanowire_metadata = MetadataCatalog.get(nanowireStr + "_Train")

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(setupOptions.modelPath)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # (default: 512, balloon test used 128)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = setupOptions.numClasses  # only has one class (VerticalNanowires)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000  # Increased from COCO default, should never have more than 2000 wires per image (default: 100)

    predictor = DefaultPredictor(cfg)

    return predictor, npImage, scaleBarMicronsPerPixel * 1000, setupOptions, nanowire_metadata


def main():

    predictor, npImage, scaleBarNMPerPixel, setupOptions, nanowire_metadata = getInstances()
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    try:
        wireMeasurementsDict = fileHandling(setupOptions.wireMeasurementsPath)
    except JSONDecodeError:
        wireMeasurementsDict = {}
    if getNakedNameFromFilePath(setupOptions.imageFilePath) not in wireMeasurementsDict:
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)] = {}

    if setupOptions.tiltAngle == 0:
        gridSize = 4
        croppedImageList = splitSingleImage(setupOptions.imageFilePath, os.getcwd(), gridSize=gridSize, saveSplitImages=False, deleteOriginalImage=False, removeDataBar=True)

        if setupOptions.parallelProcessing:
            with joblib.parallel_backend('loky'):
                with tqdm_joblib(tqdm(desc="Analyzing Instances", total=len(croppedImageList))) as progress_bar:
                    analysisOutput = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(analyzeTopDownInstance)(predictor, nanowire_metadata, scaleBarNMPerPixel, setupOptions, image) for image in croppedImageList)
        else:
            analysisOutput = []
            for image in croppedImageList:
                analysisOutput.append(analyzeTopDownInstance(predictor, nanowire_metadata, scaleBarNMPerPixel, setupOptions, image))

        totalNumVerticalWires = np.sum([entry[0] for entry in analysisOutput])
        totalNumInclinedWires = np.sum([entry[1] for entry in analysisOutput])
        totalImageAreaMicronsSq = np.sum([entry[2] for entry in analysisOutput])
        annotatedImageList = [entry[3] for entry in analysisOutput]
        wiresPerSqMicron = (totalNumVerticalWires + totalNumInclinedWires) / totalImageAreaMicronsSq

        wireDiameterListofLists = [entry[4] for entry in analysisOutput]
        compiledWireDiameterList = [element for innerList in wireDiameterListofLists for element in innerList]

        print("Analyzed Image:", getNakedNameFromFilePath(setupOptions.imageFilePath))
        print(totalNumVerticalWires, " Vertical wires, ", totalNumInclinedWires, " Inclined Wires. Together that is:", totalNumVerticalWires + totalNumInclinedWires, " Wires in ", round(totalImageAreaMicronsSq, 1), " um^2")
        print(round(wiresPerSqMicron, 3), "wires/um^2")
        print("Wire diameter list in nm: ", compiledWireDiameterList)
        print("Average wire diameter in nm:", np.average(compiledWireDiameterList))
        pyp3rclip.copy(str(compiledWireDiameterList).strip('[]'))

        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Number Vertical Wires"] = totalNumVerticalWires
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Number Inclined Wires"] = totalNumInclinedWires
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Image Area (Square Microns)"] = totalImageAreaMicronsSq
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Wires Per Square Micron"] = wiresPerSqMicron
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Wire Diameter List (nm)"] = compiledWireDiameterList
        wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Average Wire Diameter (nm)"] = np.average(compiledWireDiameterList)

        if setupOptions.showPlots:
            collageImage = create_collage(annotatedImageList, 0.02)
            collageImagePath = getNakedNameFromFilePath(setupOptions.imageFilePath)+"_annotated.jpg"
            collageImage.save(collageImagePath)
            os.system("xdg-open " + collageImagePath)

    else:
        outputs = predictor(npImage)
        maskDict, boundingBoxPolyDict, numInstances = getMaskAndBBDicts(outputs)
        if setupOptions.parallelProcessing:
            with joblib.parallel_backend('multiprocessing'):
                with tqdm_joblib(tqdm(desc="Analyzing Instances", total=numInstances)) as progress_bar:
                    analysisOutput = joblib.Parallel(n_jobs=multiprocessing.cpu_count())(joblib.delayed(analyzeSingleTiltInstance)(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions) for instanceNumber in range(numInstances))
        else:
            analysisOutput = []
            for instanceNumber in range(numInstances):
                analysisOutput.append(analyzeSingleTiltInstance(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions))

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

        allLineLengthList = [scaleBarNMPerPixel * entry[1] for entry in analysisOutput if entry[0]]
        lineStdList = [scaleBarNMPerPixel * entry[2] for entry in analysisOutput if entry[0]]
        lineAvgList = [scaleBarNMPerPixel * entry[3] for entry in analysisOutput if entry[0]]
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

        if setupOptions.isVerticalSubSection:

            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Average Width (nm)"] = averageMeasValue
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Width Standard Deviation (nm)"] = np.std(finalLineAvgList)
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Number Width Measurements"] = len(finalLineAvgList)
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Wire Width Measurements List (nm)"] = finalLineAvgList

            IndividualWireWidthsStr = "Individual Wire Widths"
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr] = {}

            for wireNumber in range(len(finalLineAvgList)):
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr][wireNumber] = {}
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr][wireNumber]["Average Width (nm)"] = finalLineAvgList[wireNumber]
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr][wireNumber]["Width Standard Deviation (nm)"] = finalLineStdList[wireNumber]
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr][wireNumber]["Number Width Measurements"] = len(finalAllLineLengthList[wireNumber])
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireWidthsStr][wireNumber]["Individual Width Measurements (nm)"] = finalAllLineLengthList[wireNumber]
        else:
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Average Length (nm)"] = averageMeasValue
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Length Standard Deviation (nm)"] = np.std(finalLineAvgList)
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Number Length Measurements"] = len(finalLineAvgList)
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)]["Wire Length Measurements List (nm)"] = finalLineAvgList

            IndividualWireLengthsStr = "Individual Wire Lengths"
            wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr] = {}
            for wireNumber in range(len(finalLineAvgList)):
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr][wireNumber] = {}
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr][wireNumber]["Average Length (nm)"] = finalLineAvgList[wireNumber]
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr][wireNumber]["Length Standard Deviation (nm)"] = finalLineStdList[wireNumber]
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr][wireNumber]["Number Length Measurements"] = len(finalAllLineLengthList[wireNumber])
                wireMeasurementsDict[getNakedNameFromFilePath(setupOptions.imageFilePath)][IndividualWireLengthsStr][wireNumber]["Individual Length Measurements (nm)"] = finalAllLineLengthList[wireNumber]

        # print(finalLineAvgList)
        # print(finalLineStdList)
        print("Analyzed Image:", getNakedNameFromFilePath(setupOptions.imageFilePath))
        print("Overall Average Size (with std dev): {:.0f} nm with random standard deviation of {:.0f} nm".format(averageMeasValue, np.std(finalLineAvgList)))
        print("Number of Measurements: ", len(finalLineAvgList))
        roundedFinalLineAvgList = [int(round(lengthMeasurement, 0)) for lengthMeasurement in finalLineAvgList]
        print("Wire size list in nm: ", roundedFinalLineAvgList)
        pyp3rclip.copy(str(roundedFinalLineAvgList).strip('[]'))


    # Copy the original wireMeasurements file to a backup before overwriting for safety, then delete the copy
    wireMeasurementsCopyPathName, wireMeasurementsCopyPathExtension = os.path.splitext(setupOptions.wireMeasurementsPath)
    wireMeasurementsCopyPathName = wireMeasurementsCopyPathName + 'backup' + wireMeasurementsCopyPathExtension
    shutil.copy2(setupOptions.wireMeasurementsPath, wireMeasurementsCopyPathName)

    with open(setupOptions.wireMeasurementsPath, 'w') as handle:
        json.dump(wireMeasurementsDict, handle, cls=NumpyEncoder)
    os.remove(wireMeasurementsCopyPathName)


if __name__ == "__main__":
    main()