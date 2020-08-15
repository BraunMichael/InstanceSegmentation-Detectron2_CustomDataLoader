import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
# from Utility.CropScaleSave import importRawImageAndScale, getNakedNameFromFilePath
# from Utility.AnalyzeOutputUI import SetupOptions
from Utility.Utilities import *
from Utility.ImageGridding import splitSingleImage


# @profile
def analyzeTopDownInstances(mask, npImage, outputs, nanowire_metadata, scaleBarNMPerPixel, setupOptions: SetupOptions):
    if setupOptions.showPlots:
        fig, ax = plt.subplots(figsize=(10, 8))
        print(setupOptions.imageFilePath)
        try:
            visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
        except IndexError:
            npImage = np.expand_dims(npImage, axis=2)
            visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
        out = visualizerNP.draw_instance_predictions(outputs["instances"].to("cpu"))
        ax.imshow(out.get_image()[:, :, ::-1])
        plt.show(block=True)

    croppedImageList = splitSingleImage(setupOptions.imageFilePath, _, 4, saveSplitImages=False, deleteOriginalImage=False)
    # TODO: go through each image and do analysis, find edge instances and count as 0.5
    imageWidth = mask.shape[1]
    imageHeight = mask.shape[0]
    imageAreaMicronsSq = imageWidth * (scaleBarNMPerPixel / 1000) * imageHeight * (scaleBarNMPerPixel / 1000)
    outputClasses = outputs['instances'].pred_classes
    classesNums, classCounts = np.unique(outputClasses, return_counts=True)
    outputClassesNumDict = dict(zip(classesNums, classCounts))

    verticalWireClass = 2
    mergedWireClass = 1
    inclinedWireClass = 0
    try:
        numVerticalWires = outputClassesNumDict[verticalWireClass]
    except KeyError:
        numVerticalWires = 0
    try:
        numMergedWires = outputClassesNumDict[mergedWireClass]
    except KeyError:
        numMergedWires = 0
    try:
        numInclinedWires = outputClassesNumDict[inclinedWireClass]
    except KeyError:
        numInclinedWires = 0

    print(numVerticalWires, " Vertical wires, ", numMergedWires, " Merged wires, ", numInclinedWires, " Inclined Wires")
    print(numVerticalWires + 2 * numMergedWires + numInclinedWires, " Wires in ", imageAreaMicronsSq, " um^2")
    wiresPerSqMicron = (numVerticalWires + 2 * numMergedWires + numInclinedWires) / imageAreaMicronsSq
    print(wiresPerSqMicron, "wires/um^2")



