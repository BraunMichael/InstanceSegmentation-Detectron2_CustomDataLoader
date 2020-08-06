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


def bboxToPoly(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


# @profile
def analyzeSingleTopDownInstance(maskDict, boundingBoxPolyDict, instanceNumber, setupOptions: SetupOptions):
    if not setupOptions.parallelProcessing:
        print("Working on instance number: ", instanceNumber)

    mask = maskDict[instanceNumber]
    imageWidth = mask.shape[1]
    imageHeight = mask.shape[0]

    measLineList = []
    lineLengthList = []


    return measLineList, filteredLineLengthList, lineStd, lineAvg, maskAngle




