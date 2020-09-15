import sys
import os
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.strtree import STRtree
from shapely import affinity
from skimage.measure import label, regionprops
from collections import OrderedDict
from polylidarutil import (plot_points, plot_polygons, get_point)
from polylidar import extractPolygons
# from Utility.CropScaleSave import importRawImageAndScale, getNakedNameFromFilePath
# from Utility.AnalyzeOutputUI import SetupOptions
from Utility.Utilities import *


# @profile
def centerXPercentofWire(npMaskFunc, setupOptions: SetupOptions):
    # TODO: This is the limiting factor at this point in speed, but it is entirely in the skimage (label_image and allRegionProperties lines) and polylidar (polygonsList line) calls
    assert 0 <= setupOptions.centerFractionToMeasure <= 1, "Percent size of section has to be between 0 and 1"
    assert isinstance(setupOptions.isVerticalSubSection, bool), "isVerticalSubSection must be a boolean, True if you want a vertical subsection, False if you want a horizontal subsection"
    label_image = label(npMaskFunc, connectivity=1)
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
        ymin, xmin, ymax, xmax = region.bbox  # may not need this line

        # maskCords as [row, col] ie [y, x]
        maskCoords = np.array(region.coords)
        flippedMaskCoords = maskCoords.copy()
        flippedMaskCoords[:, 0], flippedMaskCoords[:, 1] = flippedMaskCoords[:, 1], flippedMaskCoords[:, 0].copy()
        maskAngle = np.rad2deg(region.orientation)

        polygonsList = extractPolygons(flippedMaskCoords)
        # assert len(polygonsList) == 1, "There was more than 1 polygon extracted from extractPolygons."
        # Just take the first polygon no matter what
        shell_coords = [get_point(pi, flippedMaskCoords) for pi in polygonsList[0].shell]
        maskPolygon = Polygon(shell=shell_coords)
        if setupOptions.plotPolylidar and not setupOptions.parallelProcessing:
            fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)
            # plot points
            plot_points(flippedMaskCoords, ax)
            # plot polygons...doesn't use 2nd argument
            plot_polygons(polygonsList, 0, flippedMaskCoords, ax)
            plt.axis('equal')
            plt.show()

        # Scale the width/height and then skew the squared bounding box by skimage ellipse fitted mask angle about skimage mask centroid
        # This should ensure a measurement line passes through the centroid, at the ellipse angle, and never starts within the mask itself
        # This also means the lengths measured are the real lengths, don't need to do trig later
        centroidCoords = region.centroid
        if setupOptions.isVerticalSubSection:
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, 1, setupOptions.centerFractionToMeasure)
            # Coords are row, col ie (y, x)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, ys=-maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        else:
            # Coords are row, col ie (y, x)
            scaledBoundingBoxPoly = affinity.scale(maskPolygon.envelope, setupOptions.centerFractionToMeasure, 1)
            subBoundingBoxPoly = affinity.skew(scaledBoundingBoxPoly.envelope, xs=maskAngle, origin=(centroidCoords[1], centroidCoords[0]))
        outputSubMaskPoly = maskPolygon.intersection(subBoundingBoxPoly)

        if setupOptions.showBoundingBoxPlots and not setupOptions.parallelProcessing:
            # Blue rectangle is standard bounding box
            # Red rectangle is rotated bounding box from MinimumBoundingBox
            # Multicolored points are either standard (setupOptions.isVerticalSubsection=True) or rotated bounding box (setupOptions.isVerticalSubsection=False)
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111)
            ax.imshow(npMaskFunc)
            r1 = mpatches.Rectangle((xmin, ymax), xmax - xmin, -(ymax - ymin), fill=False, edgecolor="blue", alpha=1, linewidth=1)

            ax.axis('equal')
            ax.add_patch(r1)

            # 5, since we have 4 points for a rectangle but don't want to have 1st = 4th
            phi = -1 * np.linspace(0, 2*np.pi, 5)
            rgb_cycle = np.vstack((.5 * (1. + np.cos(phi)), .5 * (1. + np.cos(phi + 2 * np.pi / 3)), .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T
            plt.scatter(subBoundingBoxPoly.exterior.coords.xy[0][:-1], subBoundingBoxPoly.exterior.coords.xy[1][:-1], c=rgb_cycle[:4])
            plt.plot(subBoundingBoxPoly.exterior.coords.xy[0], subBoundingBoxPoly.exterior.coords.xy[1])
            plt.autoscale()
            plt.show()

        return outputSubMaskPoly, subBoundingBoxPoly, maskAngle
    # else:
    return None, None, None


def bboxToPoly(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def isValidLine(strTree, instanceBoxCoords, imageHeight, instanceLine):
    # Check if line is contained in a different bounding box, need to check which instance is in front (below)
    # Don't check for the current instanceNum (key in boundingBoxDict)
    validForInstanceList = []
    bottomLeft, bottomRight, topLeft, topRight = instanceBoxCoords
    lineIntersectingBoxes = strTree.query(instanceLine)
    if lineIntersectingBoxes:
        for bbox in lineIntersectingBoxes:
            # This is faster than doing a index lookup into a list of bounding box polygons then coords
            bottomLeftCheck, bottomRightCheck, topLeftCheck, topRightCheck = getXYFromPolyBox(bbox)
            imageBottom = imageHeight
            instanceBottom = bottomLeft[1]
            checkInstanceBottom = bottomLeftCheck[1]
            if abs(imageBottom - instanceBottom) < 20:
                # the box of interest is too close to the bottom
                instanceTop = topLeft[1]
                checkInstanceTop = topLeftCheck[1]
                if instanceTop < checkInstanceTop:
                    # Good assumption that instance of interest is in front, since all wires ~same length and the top is lower than the check instance
                    validForInstanceList.append(True)
                else:
                    # Maybe need to add more checks if too many lines are being eliminated
                    # intersectingMask = maskDict[checkNumber]
                    validForInstanceList.append(False)
            elif instanceBottom > checkInstanceBottom:
                # the instance of interest is lower in the image, thus in front due to substrate tilt
                validForInstanceList.append(True)
            else:
                validForInstanceList.append(False)
    else:
        validForInstanceList.append(True)

    if all(validForInstanceList):
        return True
    # else:
    return False


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


# @profile
def longestLineAndLengthInPolygon(maskPolygon, lineTest):
    # TODO: This is slow, just the lineTest.intersection line
    testSegments = lineTest.intersection(maskPolygon)  # without .boundary, get the lines immediately
    outputLine = None
    LineLength = None
    if isinstance(testSegments, LineString):
        if not testSegments.is_empty:
            # The line is entirely inside the mask polygon
            LineLength = testSegments.length
            outputLine = testSegments
    elif isinstance(testSegments, MultiLineString):
        # The line crosses the boundary of the mask polygon
        LineLength = 0
        for segment in testSegments:
            if segment.length > LineLength:
                LineLength = segment.length
                outputLine = segment
    return outputLine, LineLength


def getLinePoints(startXY, endXY):
    startPoint = Point(startXY)
    endPoint = Point(endXY)
    lineOfInterest = LineString([startPoint, endPoint])

    if sys.version_info < (3, 7):
        xyPoints = OrderedDict()
    else:
        # in Python 3.7 and newer dicts are ordered by default
        xyPoints = {}
    for pos in range(int(np.ceil(lineOfInterest.length)) + 1):
        interpolatedPoint = lineOfInterest.interpolate(pos).coords[0]
        roundedPoint = map(round, interpolatedPoint)
        xyPoints[tuple(roundedPoint)] = None
    return xyPoints


# @profile
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


# @profile
def analyzeSingleTiltInstance(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions: SetupOptions):
    if not setupOptions.parallelProcessing:
        print("Working on instance number: ", instanceNumber)

    mask = maskDict[instanceNumber]
    imageWidth = mask.shape[1]
    imageHeight = mask.shape[0]

    measLineList = []
    lineLengthList = []
    filteredLineLengthList = []
    lineStd = None
    lineAvg = None

    outputSubMaskPoly, subBoundingBoxPoly, maskAngle = centerXPercentofWire(mask, setupOptions)
    if outputSubMaskPoly is not None:
        if -5 < maskAngle < 5:
            strTree = STRtree([poly for i, poly in boundingBoxPolyDict.items() if i != instanceNumber])
            subStrTree = STRtree(strTree.query(boundingBoxPolyDict[instanceNumber]))

            bottomLeft, bottomRight, topLeft, topRight = getXYFromPolyBox(subBoundingBoxPoly)
            instanceBoxCoords = getXYFromPolyBox(boundingBoxPolyDict[instanceNumber])

            if not isEdgeInstance(imageWidth, imageHeight, instanceBoxCoords, setupOptions.isVerticalSubSection):
                if setupOptions.isVerticalSubSection:
                    lineStartPoints = getLinePoints(bottomLeft, topLeft)  # Left line
                    lineEndPoints = getLinePoints(bottomRight, topRight)  # Right line
                else:
                    lineStartPoints = getLinePoints(bottomLeft, bottomRight)  # Bottom line
                    lineEndPoints = getLinePoints(topLeft, topRight)  # Top line

                for startPoint, endPoint in zip(lineStartPoints, lineEndPoints):
                    instanceLine = LineString([startPoint, endPoint])
                    longestLine, lineLength = longestLineAndLengthInPolygon(outputSubMaskPoly, instanceLine)
                    if longestLine is not None:
                        if isValidLine(subStrTree, instanceBoxCoords, imageHeight, longestLine):
                            measLineList.append(longestLine)
                            lineLengthList.append(lineLength)
                if len(lineLengthList) > 2:
                    # The first and last lines sometimes have issues, remove them
                    measLineList = measLineList[1:-1]
                    lineLengthList = np.asarray(lineLengthList[1:-1])
                    if setupOptions.isVerticalSubSection:
                        filteredLineLengthList = lineLengthList

                    else:
                        filteredLineLengthList = lineLengthList[lineLengthList > (np.max(lineLengthList)-0.5*np.std(lineLengthList))]
                        measLineList = [measLineListItem for measLineListItem, lineLength in zip(measLineList, lineLengthList) if lineLength > (np.max(lineLengthList) - 0.5 * np.std(lineLengthList))]

                    if len(filteredLineLengthList) == 2:
                        lineStd = np.std(filteredLineLengthList, ddof=1)
                    elif len(filteredLineLengthList) == 1:
                        lineStd = 0
                    else:
                        lineStd = np.std(filteredLineLengthList, ddof=0)
                    lineAvg = np.mean(filteredLineLengthList)
                else:
                    # else there are no valid lines
                    measLineList = []

    return measLineList, filteredLineLengthList, lineStd, lineAvg, maskAngle




