import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from skimage.measure import label, regionprops
from detectron2.utils.visualizer import Visualizer
from Utility.Utilities import *


# @profile
def analyzeTopDownInstance(predictor, nanowire_metadata, scaleBarNMPerPixel, setupOptions: SetupOptions, image):
    npImage = np.array(image)
    if npImage.ndim < 3:
        if npImage.ndim == 2:
            # Assuming black and white image, just copy to all 3 color channels
            npImage = np.repeat(npImage[:, :, np.newaxis], 3, axis=2)
        else:
            print('The imported rawImage is 1 dimensional for some reason, check it out.')
            quit()
    outputs = predictor(npImage)
    annotatedImage = None

    # fig, ax = plt.subplots(figsize=(10, 8))
    # print(setupOptions.imageFilePath)
    try:
        visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
    except IndexError:
        npImage = np.expand_dims(npImage, axis=2)
        visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
    out = visualizerNP.draw_instance_predictions(outputs["instances"].to("cpu"))
    annotatedImage = out.get_image()[:, :, ::-1]
    # ax.imshow(annotatedImage)
    # plt.show(block=True)

    (imageWidth, imageHeight) = image.size
    imageAreaMicronsSq = imageWidth * (scaleBarNMPerPixel / 1000) * imageHeight * (scaleBarNMPerPixel / 1000)

    maskDict, boundingBoxPolyDict, numInstances = getMaskAndBBDicts(outputs)
    outputClasses = outputs['instances'].pred_classes

    verticalWireClass = 0
    inclinedWireClass = 1

    numVerticalWires = 0
    numInclinedWires = 0

    wireDiameterList = []
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
            if wireFraction == 1:  #is not a side or corner instance
                label_image = label(maskDict[instanceNumber], connectivity=1)
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
                    wireArea = region.area
                    wireDiameter = 2*np.sqrt(wireArea/pi)*scaleBarNMPerPixel
                    wireDiameterList.append(wireDiameter)

        elif outputClasses[instanceNumber] == inclinedWireClass:
            numInclinedWires += wireFraction
        else:
            print('Too many classes')
            quit()

    return numVerticalWires, numInclinedWires, imageAreaMicronsSq, annotatedImage, wireDiameterList




