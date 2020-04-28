import os
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import easygui
from PIL import Image
from tkinter import Tk
from tkinter import filedialog


def longestContinuousLengthPerLine(whitePixelsDict):
    longestContinuousLengthPerLineDict = {}

    # key is line number, value is list of tuples of (line start pixel, line end pixel)
    longLinesPerLineDict = {}

    for line in whitePixelsDict:
        longestContinuousLengthPerLineDict[line] = 1
        continuousLength = 1

        pixel = 0

        while pixel < (len(whitePixelsDict[line]) - 1):
            pixelPositionDifference = whitePixelsDict[line][pixel + 1] - whitePixelsDict[line][pixel]
            if pixelPositionDifference == 1:
                continuousLength += 1
            else:
                if continuousLength > 100:
                    if line not in longLinesPerLineDict:
                        longLinesPerLineDict[line] = []
                    longLinesPerLineDict[line].append(
                        (whitePixelsDict[line][pixel] - continuousLength + 1, whitePixelsDict[line][pixel]))
                if continuousLength > longestContinuousLengthPerLineDict[line]:
                    longestContinuousLengthPerLineDict[line] = continuousLength
                continuousLength = 1

            # check if we are on penultimate pixel, ie next pixel is last one and will kill while loop
            if (pixel + 1) == (len(whitePixelsDict[line]) - 1):
                # Even if line is continuous, there is no next loop, need to add to dicts now!
                if pixelPositionDifference == 1:
                    if continuousLength > 100:
                        if line not in longLinesPerLineDict:
                            longLinesPerLineDict[line] = []
                        longLinesPerLineDict[line].append(
                            (whitePixelsDict[line][pixel + 1] - continuousLength + 1, whitePixelsDict[line][pixel + 1]))
                    if continuousLength > longestContinuousLengthPerLineDict[line]:
                        longestContinuousLengthPerLineDict[line] = continuousLength
                    continuousLength = 1

            pixel += 1
    return longestContinuousLengthPerLineDict, longLinesPerLineDict


def whitePixels(image, horizontalLine_truefalse):
    (imageWidth, imageHeight) = image.size
    whitePixelsDict = {}
    for y in range(0, imageHeight):
        for x in range(0, imageWidth):
            if image.getpixel((x, y)) > 200:
                if horizontalLine_truefalse is True:
                    if y not in whitePixelsDict:
                        whitePixelsDict[y] = []
                    whitePixelsDict[y].append(x)
                else:
                    if x not in whitePixelsDict:
                        whitePixelsDict[x] = []
                    whitePixelsDict[x].append(y)
    return whitePixelsDict


def onselect(eclick, erelease):
    if eclick.ydata > erelease.ydata:
        eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
    if eclick.xdata > erelease.xdata:
        eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
    ax.set_ylim(erelease.ydata, eclick.ydata)
    ax.set_xlim(eclick.xdata, erelease.xdata)
    fig.canvas.draw()


def whiteLinePixelLocationList(rawImage, horizontalLine_truefalse):
    whiteLinesList = []
    (imageWidth, imageHeight) = rawImage.size
    whitePixelsDict = whitePixels(rawImage, horizontalLine_truefalse)
    (longestContinuousLengthPerLineDict, _) = longestContinuousLengthPerLine(whitePixelsDict)
    # Assumes if longer than 60% of width/height then it's a line of interest
    for line in longestContinuousLengthPerLineDict:
        if horizontalLine_truefalse is True:
            if longestContinuousLengthPerLineDict[line] > 0.6 * imageWidth:
                whiteLinesList.append(line)
        else:
            if longestContinuousLengthPerLineDict[line] > 0.6 * imageHeight:
                whiteLinesList.append(line)

    return whiteLinesList


# add logic to find white line bounding box of scale bar
def scaleBarProcessing(filename):
    global fig
    global ax
    fig, ax = plt.subplots()
    rawImage = Image.open(filename)
    (rawImageWidth, rawImageHeight) = rawImage.size
    # .crop(left upper right lower)
    rawImageHeightOffset = rawImageHeight * 0.75
    rawImageWidthOffset = rawImageWidth * 0.5
    reducedRawImage = rawImage.crop((rawImageWidthOffset, rawImageHeightOffset, rawImageWidth, rawImageHeight))

    # Find top of databar
    horizontalLine_truefalse = True
    whiteRowList = whiteLinePixelLocationList(reducedRawImage, horizontalLine_truefalse)

    # Don't need this anymore, only taking the bottom section of the image with reducedRawImage
    # for index in range(0, len(whiteRowList)-1):
    # Ignore lines in the top half of the image, basically removes very top line
    #	if whiteRowList[index] < (imageHeight/2):
    #		whiteRowList.pop(index)

    dataBarPixelRow = whiteRowList[0]
    dataBarPixelRow_OffsetCorrected = dataBarPixelRow + rawImageHeightOffset

    # Find edge of box of scalebar (closest line to right side that isn't the right side)

    # Don't need this anymore, switched to reducedRawImage
    # intermediateImage = rawImage.crop((0, dataBarPixelRow, imageWidth, imageHeight))

    (imageWidth, imageHeight) = reducedRawImage.size
    intermediateImage = reducedRawImage.crop((0, dataBarPixelRow, imageWidth, imageHeight))

    horizontalLine_truefalse = False
    whiteColumnList = whiteLinePixelLocationList(intermediateImage, horizontalLine_truefalse)
    # Assume the rightmost (last) line is the edge of the scalebar box, we want the next one
    scalebarColumn = whiteColumnList[-2] + rawImageWidthOffset

    arr = np.asarray(rawImage)
    plt_image = plt.imshow(arr)
    ax.set_ylim(rawImageHeight, dataBarPixelRow_OffsetCorrected)
    ax.set_xlim(scalebarColumn, rawImageWidth)
    rs = widgets.RectangleSelector(ax, onselect, drawtype='box',
                                   rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
    plt.show()
    scaleBarValue = easygui.integerbox(msg="After satisfied with cropping, enter the scale bar size in microns",
                                       title="Get scale bar size", upperbound=100000)

    (xmin, xmax) = ax.get_xlim()
    (ymax, ymin) = ax.get_ylim()
    plt.close()

    croppedImage = rawImage.crop((xmin, ymin, xmax, ymax))
    (cropWidth, cropHeight) = croppedImage.size

    horizontalLine_truefalse = True
    whitePixelsDict = whitePixels(croppedImage, horizontalLine_truefalse)
    (longestContinuousLengthPerLineDict, longLinesPerLineDict) = longestContinuousLengthPerLine(whitePixelsDict)

    while True:
        longestContinuousRow = max(longestContinuousLengthPerLineDict.keys(),
                                   key=(lambda k: longestContinuousLengthPerLineDict[k]))
        if longestContinuousLengthPerLineDict[longestContinuousRow] > 0.6 * cropWidth:
            # Probably a continuous line and not the actual scale bar (which is cut in half by the actual value)
            # print("Disqualified row: ",longestContinuousRow+ymin)
            longestContinuousLengthPerLineDict[longestContinuousRow] = 0
        else:

            # Sort by length of line (largest to smallest), break ties with start point of each tuple (smallest to largest)
            sortedPossibleScaleBarLines = sorted(longLinesPerLineDict[longestContinuousRow],
                                                 key=lambda x: (x[1] - x[0], -x[0]), reverse=True)
            # Scale bar is actually split into 2 halves, assuming roughly equal, give a 20 pixel buffer, but they really should be close to same length if its actually the scale bar

            longestLineLength = sortedPossibleScaleBarLines[0][1] - sortedPossibleScaleBarLines[0][0]
            nextLongestLineLength = sortedPossibleScaleBarLines[1][1] - sortedPossibleScaleBarLines[1][0]

            if abs(longestLineLength - nextLongestLineLength) < 20:
                # Calculate length by subtracting the last point of the right from the first of the left line. If statement to make sure the order of the lines is correct!
                if sortedPossibleScaleBarLines[1][1] > sortedPossibleScaleBarLines[0][0]:
                    scaleBarStartPixel_OffsetCorrected = sortedPossibleScaleBarLines[0][0] + xmin
                    scaleBarEndPixel_OffsetCorrected = sortedPossibleScaleBarLines[1][1] + xmin
                else:
                    scaleBarStartPixel_OffsetCorrected = sortedPossibleScaleBarLines[1][0] + xmin
                    scaleBarEndPixel_OffsetCorrected = sortedPossibleScaleBarLines[0][1] + xmin
                print("Row: ", longestContinuousRow + ymin, " With scalebar edge pixels: (",
                      scaleBarStartPixel_OffsetCorrected, ", ", scaleBarEndPixel_OffsetCorrected, ")")
                scaleBarWidthPixels = scaleBarEndPixel_OffsetCorrected - scaleBarStartPixel_OffsetCorrected
            break
    scaleBarMicronsPerPixel = scaleBarValue / scaleBarWidthPixels
    rawImage.close()
    croppedImage.close()
    print("Scale bar is ", scaleBarWidthPixels, " pixels across. Total width of cropped area is: ", cropWidth)
    print("Scale Bar Microns Per Pixel is: ", scaleBarMicronsPerPixel)

    return (scaleBarMicronsPerPixel, dataBarPixelRow_OffsetCorrected)


def getScaleandDataBarDicts(fileNames):
    scaleBarDict = {}
    dataBarPixelRowDict = {}
    for name in fileNames:
        (scaleBarMicronsPerPixel, dataBarPixelRow) = scaleBarProcessing(name)
        scaleBarDict[name] = scaleBarMicronsPerPixel
        dataBarPixelRowDict[name] = dataBarPixelRow
    return (scaleBarDict, dataBarPixelRowDict)


root = Tk()
root.withdraw()
filesFolder = filedialog.askdirectory(initialdir='E:\Google Drive\Research SEM')
(dirpath, dirnames, rawFileNames) = next(os.walk(filesFolder))
fileNames = []

for name in rawFileNames:
    # if name.endswith(('.tiff','.tif','jpg','.jpeg')) and name.find('cropped') == -1:
    if name.endswith(('jpg', '.jpeg')) and name.find('cropped') == -1:
        fileNames.append(os.path.join(dirpath, name))
    if name.endswith(('.tiff', '.tif')):
        print("Warning, tiffs do not work yet")

(scaleBarDict, dataBarPixelRowDict) = getScaleandDataBarDicts(fileNames)

for name in fileNames:
    # processing for each file here

    scaleBarMicronsPerPixel = scaleBarDict[name]
    rawImage = Image.open(name)
    (imageWidth, imageHeight) = rawImage.size

    # This is the white line closest to the middle of the image in the bottom half of the image. Should be top of databar
    dataBarPixelRow = dataBarPixelRowDict[name]
    croppedImage = rawImage.crop((0, 0, imageWidth, dataBarPixelRow - 1))
    databarImage = rawImage.crop((0, dataBarPixelRow, imageWidth, imageHeight))
    rawImage.close()

    fileTypeEnding = name[name.rfind('.'):]
    croppedFileName = name.replace(fileTypeEnding, '_cropped' + fileTypeEnding)
    croppedImage.save(croppedFileName)









