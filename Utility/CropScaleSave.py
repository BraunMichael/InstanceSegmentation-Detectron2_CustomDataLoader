import os
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import easygui
import shutil
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

        while pixel < (len(whitePixelsDict[line])-1):
            pixelPositionDifference = whitePixelsDict[line][pixel+1]-whitePixelsDict[line][pixel]
            if pixelPositionDifference == 1:
                continuousLength += 1
            else:
                if continuousLength > 100:
                    if line not in longLinesPerLineDict:
                        longLinesPerLineDict[line] = []
                    longLinesPerLineDict[line].append((whitePixelsDict[line][pixel]-continuousLength+1, whitePixelsDict[line][pixel]))
                if continuousLength > longestContinuousLengthPerLineDict[line]:
                    longestContinuousLengthPerLineDict[line] = continuousLength
                continuousLength = 1

            # check if we are on penultimate pixel, ie next pixel is last one and will kill while loop
            if (pixel+1) == (len(whitePixelsDict[line])-1):
                # Even if line is continuous, there is no next loop, need to add to dicts now!
                if pixelPositionDifference == 1:
                    if continuousLength > 100:
                        if line not in longLinesPerLineDict:
                            longLinesPerLineDict[line] = []
                        longLinesPerLineDict[line].append((whitePixelsDict[line][pixel+1]-continuousLength+1, whitePixelsDict[line][pixel+1]))
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


def scaleBarProcessing(filename, scaleBarMicronsPerPixelDict, replaceScaleEntry):
    rawImage = Image.open(filename)
    (rawImageWidth, rawImageHeight) = rawImage.size
    rawImageHeightOffset = rawImageHeight*0.75
    rawImageWidthOffset = rawImageWidth*0.5
    # crop(left upper right lower)
    reducedRawImage = rawImage.crop((rawImageWidthOffset, rawImageHeightOffset, rawImageWidth, rawImageHeight))

    # Find top of databar
    horizontalLine_truefalse = True
    whiteRowList = whiteLinePixelLocationList(reducedRawImage, horizontalLine_truefalse)

    dataBarPixelRow = whiteRowList[0]
    dataBarPixelRow_OffsetCorrected = dataBarPixelRow + rawImageHeightOffset

    nakedFileName = getNakedNameFromFilePath(filename)
    if replaceScaleEntry or nakedFileName not in scaleBarMicronsPerPixelDict:
        global fig
        global ax
        fig, ax = plt.subplots()
        # Find edge of box of scalebar (closest line to right side that isn't the right side)

        (imageWidth, imageHeight) = reducedRawImage.size
        intermediateImage = reducedRawImage.crop((0, dataBarPixelRow, imageWidth, imageHeight))

        horizontalLine_truefalse = False
        whiteColumnList = whiteLinePixelLocationList(intermediateImage, horizontalLine_truefalse)
        # Assume the rightmost (last) line is the edge of the scalebar box, we want the next one
        scalebarColumn = whiteColumnList[-2] + rawImageWidthOffset

        arr = np.asarray(rawImage)
        plt.imshow(arr)
        ax.set_ylim(rawImageHeight, dataBarPixelRow_OffsetCorrected)
        ax.set_xlim(scalebarColumn, rawImageWidth)
        widgets.RectangleSelector(ax, onselect, drawtype='box',  rectprops=dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True))
        plt.show()
        scaleBarValue = easygui.integerbox(msg="After satisfied with cropping, enter the scale bar size in microns", title="Get scale bar size", upperbound=100000)

        (xmin, xmax) = ax.get_xlim()
        (ymax, ymin) = ax.get_ylim()
        plt.close('all')

        croppedImage = rawImage.crop((xmin, ymin, xmax, ymax))
        (cropWidth, cropHeight) = croppedImage.size

        horizontalLine_truefalse = True
        whitePixelsDict = whitePixels(croppedImage, horizontalLine_truefalse)
        (longestContinuousLengthPerLineDict, longLinesPerLineDict) = longestContinuousLengthPerLine(whitePixelsDict)
        scaleBarWidthPixels = 0
        while True:
            longestContinuousRow = max(longestContinuousLengthPerLineDict.keys(), key=(lambda k: longestContinuousLengthPerLineDict[k]))
            if longestContinuousLengthPerLineDict[longestContinuousRow] > 0.6 * cropWidth:
                # Probably a continuous line and not the actual scale bar (which is cut in half by the actual value)
                longestContinuousLengthPerLineDict[longestContinuousRow] = 0
            else:
                # Sort by length of line (largest to smallest), break ties with start point of each tuple (smallest to largest)
                sortedPossibleScaleBarLines = sorted(longLinesPerLineDict[longestContinuousRow], key=lambda x: (x[1]-x[0], -x[0]), reverse=True)
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
                    print("Row: ", longestContinuousRow+ymin, " With scalebar edge pixels: (", scaleBarStartPixel_OffsetCorrected, ", ", scaleBarEndPixel_OffsetCorrected, ")")
                    scaleBarWidthPixels = scaleBarEndPixel_OffsetCorrected - scaleBarStartPixel_OffsetCorrected
                break
        assert scaleBarWidthPixels > 0, "Could not find a scale bar, maybe something is different about the input file databar format or white box surrounding it?"
        scaleBarMicronsPerPixel = scaleBarValue/scaleBarWidthPixels
        croppedImage.close()
        print("Scale bar is ", scaleBarWidthPixels, " pixels across. Total width of cropped area is: ", cropWidth)
        scaleBarMicronsPerPixelDict[getNakedNameFromFilePath(filename)] = scaleBarMicronsPerPixel
    else:  # use existing scaleBarMicronsPerPixel from dict
        scaleBarMicronsPerPixel = scaleBarMicronsPerPixelDict[nakedFileName]

    rawImage.close()
    print("Scale Bar Microns Per Pixel is: ", scaleBarMicronsPerPixel)

    return scaleBarMicronsPerPixelDict, dataBarPixelRow_OffsetCorrected


def getScaleandDataBarDicts(fileNames, scaleBarMicronsPerPixelDict, replaceScaleEntry):
    dataBarPixelRowDict = {}
    for name in fileNames:
        (scaleBarMicronsPerPixelDict, dataBarPixelRow) = scaleBarProcessing(name, scaleBarMicronsPerPixelDict, replaceScaleEntry)
        dataBarPixelRowDict[name] = dataBarPixelRow
    return scaleBarMicronsPerPixelDict, dataBarPixelRowDict


def getScaleDictFromFile(scaleBarDictFileFunc):
    scaleBarMicronsPerPixelDict = {}
    with open(scaleBarDictFileFunc, 'r') as file:
        for line in file:
            splitLine = line.split()
            # print(splitLine)
            if splitLine[0] not in scaleBarMicronsPerPixelDict:
                scaleBarMicronsPerPixelDict[splitLine[0]] = float(splitLine[1])
            else:
                print('There are duplicates in your scaleBarMicronsPerPixelDict file, fix it')
                quit()
    return scaleBarMicronsPerPixelDict


def getNakedNameFromFilePath(name):
    head, tail = os.path.split(name)
    nakedName, fileExtension = os.path.splitext(tail)
    return nakedName


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


def importRawImageAndScale():
    root = Tk()
    root.withdraw()
    inputFileName = getFileOrDirList('file', 'Choose input image file')

    scaleBarDictFile = filedialog.askopenfilename(initialdir=r'E:\Google Drive\Research SEM', title='Choose scalebar Dict file', filetypes=[('Scalebar Dict Text File', '.txt')])
    scaleBarMicronsPerPixelDict = getScaleDictFromFile(scaleBarDictFile)

    fileNames = []
    if inputFileName.endswith(('jpg', '.jpeg')):
        fileNames.append(inputFileName)

    if inputFileName.endswith(('.tiff', '.tif')):
        print("attempting to convert tiff to png")
        imagePath = inputFileName

        fileTypeEnding = imagePath[imagePath.rfind('.'):]
        pngName = inputFileName.replace(fileTypeEnding, '.png')
        pngPath = os.path.join(dirpath, pngName)
        rawImage = Image.open(imagePath)
        npImage = ((np.array(rawImage) + 1) / 256) - 1
        visImage = Image.fromarray(np.uint8(npImage), mode='L')
        visImage.save(pngPath, 'PNG')
        fileNames.append(os.path.join(dirpath, pngName))
        # os.remove(imagePath)

    noDuplicateNames = True
    namesLeftToCheck = len(fileNames)
    nameNum = 0
    replaceScaleEntry = True
    while noDuplicateNames & (nameNum < namesLeftToCheck):
        if getNakedNameFromFilePath(fileNames[nameNum]) in scaleBarMicronsPerPixelDict:
            noDuplicateNames = False
            replaceScaleEntry = easygui.boolbox('Replace existing scale per pixel for each image or use existing values?',
                                                'Replace or Use Existing?', ['Replace', 'Use Existing'], cancel_choice='None')
            if replaceScaleEntry is None:
                replaceScaleEntry = True
        nameNum += 1

    (scaleBarMicronsPerPixelDict, dataBarPixelRowDict) = getScaleandDataBarDicts(fileNames, scaleBarMicronsPerPixelDict, replaceScaleEntry)

    # Copy the original scaleBarDictFile to a backup before overwriting for safety, then delete the copy
    scaleBarDictFileCopyName, scaleBarDictFileExtension = os.path.splitext(scaleBarDictFile)
    scaleBarDictFileCopyName = scaleBarDictFileCopyName + 'backup' + scaleBarDictFileExtension
    shutil.copy2(scaleBarDictFile, scaleBarDictFileCopyName)
    with open(scaleBarDictFile, 'w') as file:
        for key, value in scaleBarMicronsPerPixelDict.items():
            file.write('%s\t%s\n' % (getNakedNameFromFilePath(key), str(value)))
    os.remove(scaleBarDictFileCopyName)

    inputFileName = fileNames[0]

    scaleBarMicronsPerPixel = scaleBarMicronsPerPixelDict[getNakedNameFromFilePath(inputFileName)]
    rawImage = Image.open(inputFileName)
    (imageWidth, imageHeight) = rawImage.size

    # This is the white line closest to the middle of the image in the bottom half of the image. Should be top of databar
    dataBarPixelRow = dataBarPixelRowDict[inputFileName]
    croppedImage = rawImage.crop((0, 0, imageWidth, dataBarPixelRow-1))
    # databarImage = rawImage.crop((0, dataBarPixelRow, imageWidth, imageHeight))
    rawImage.close()

    fileTypeEnding = inputFileName[inputFileName.rfind('.'):]
    croppedFileName = inputFileName.replace(fileTypeEnding, '_cropped'+fileTypeEnding)
    croppedImage.save(croppedFileName)

    return croppedImage, scaleBarMicronsPerPixel
