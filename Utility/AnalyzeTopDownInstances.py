import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from Utility.Utilities import *
from Utility.ImageGridding import splitSingleImage


# @profile
def analyzeTopDownInstances(predictor, nanowire_metadata, scaleBarNMPerPixel, setupOptions: SetupOptions):
    gridSize = 4
    croppedImageList = splitSingleImage(setupOptions.imageFilePath, os.getcwd(), gridSize=gridSize, saveSplitImages=False, deleteOriginalImage=False)
    numVerticalWiresArray = np.zeros(gridSize * gridSize)
    numMergedWiresArray = np.zeros(gridSize * gridSize)
    numInclinedWiresArray = np.zeros(gridSize * gridSize)
    imageAreaMicronsSqArray = np.zeros(gridSize * gridSize)

    for imageIndex, image in enumerate(croppedImageList):
        npImage = np.array(image)
        if npImage.ndim < 3:
            if npImage.ndim == 2:
                # Assuming black and white image, just copy to all 3 color channels
                npImage = np.repeat(npImage[:, :, np.newaxis], 3, axis=2)
            else:
                print('The imported rawImage is 1 dimensional for some reason, check it out.')
                quit()
        outputs = predictor(npImage)
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

        (imageWidth, imageHeight) = image.size
        imageAreaMicronsSq = imageWidth * (scaleBarNMPerPixel / 1000) * imageHeight * (scaleBarNMPerPixel / 1000)

        _, boundingBoxPolyDict, numInstances = getMaskAndBBDicts(outputs)
        outputClasses = outputs['instances'].pred_classes

        verticalWireClass = 2
        mergedWireClass = 1
        inclinedWireClass = 0

        numVerticalWires = 0
        numMergedWires = 0
        numInclinedWires = 0
        for instanceNumber in range(numInstances):
            instanceBoxCoords = getXYFromPolyBox(boundingBoxPolyDict[instanceNumber])
            wireFraction = 1
            isLRSideInstance = False
            isTBSideInstance = False
            if isEdgeInstance(imageWidth, imageHeight, instanceBoxCoords, True):
                isLRSideInstance = True
            if isEdgeInstance(imageWidth, imageHeight, instanceBoxCoords, False):
                isTBSideInstance = True
            if isLRSideInstance and isTBSideInstance:  # It's a corner instance
                wireFraction = 0.25
            elif isLRSideInstance or isTBSideInstance:  # It's a side instance, but not a corner instance
                wireFraction = 0.5
            if outputClasses[instanceNumber] == verticalWireClass:
                numVerticalWires += wireFraction
            elif outputClasses[instanceNumber] == mergedWireClass:
                numMergedWires += wireFraction
            elif outputClasses[instanceNumber] == inclinedWireClass:
                numInclinedWires += wireFraction
            else:
                print('Too many classes')
                quit()

        numVerticalWiresArray[imageIndex] = numVerticalWires
        numMergedWiresArray[imageIndex] = numMergedWires
        numInclinedWiresArray[imageIndex] = numInclinedWires
        imageAreaMicronsSqArray[imageIndex] = imageAreaMicronsSq

    totalNumVerticalWires = numVerticalWiresArray.sum()
    totalNumMergedWires = numMergedWiresArray.sum()
    totalNumInclinedWires = numInclinedWiresArray.sum()
    totalImageAreaMicronsSq = imageAreaMicronsSqArray.sum()
    wiresPerSqMicron = (totalNumVerticalWires + 2 * totalNumMergedWires + totalNumInclinedWires) / totalImageAreaMicronsSq

    print(totalNumVerticalWires, " Vertical wires, ", totalNumMergedWires, " Merged wires, ", totalNumInclinedWires, " Inclined Wires")
    print(totalNumVerticalWires + 2 * totalNumMergedWires + totalNumInclinedWires, " Wires in ", totalImageAreaMicronsSq, " um^2")
    print(wiresPerSqMicron, "wires/um^2")
    return totalNumVerticalWires, totalNumMergedWires, totalNumInclinedWires, totalImageAreaMicronsSq, wiresPerSqMicron



