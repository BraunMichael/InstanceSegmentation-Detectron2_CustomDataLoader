import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from Utility.Utilities import *
from Utility.ImageGridding import splitSingleImage


# @profile
def analyzeTopDownInstances(predictor, nanowire_metadata, scaleBarNMPerPixel, setupOptions: SetupOptions):
    croppedImageList = splitSingleImage(setupOptions.imageFilePath, os.getcwd(), gridSize=4, saveSplitImages=False, deleteOriginalImage=False)
    for image in croppedImageList:
        # TODO: go through each image and do analysis, find edge instances and count as 0.5
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

        return numVerticalWires, numMergedWires, numInclinedWires, imageAreaMicronsSq, wiresPerSqMicron



