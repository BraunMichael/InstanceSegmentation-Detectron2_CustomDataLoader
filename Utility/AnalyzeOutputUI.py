import os
import json
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Utility.Utilities import *
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class TextValidator(object):
    def __init__(self, tkWindow, minimumScaleBarWidthMicronsValue, maximumScaleBarWidthMicronsValue, minimumTiltAngleValue, maximumTiltAngleValue, minimumCenterFractionToMeasureValue, maximumCenterFractionToMeasureValue):
        self.tkWindow = tkWindow
        self.minimumScaleBarWidthMicronsValue = minimumScaleBarWidthMicronsValue
        self.maximumScaleBarWidthMicronsValue = maximumScaleBarWidthMicronsValue
        self.minimumTiltAngleValue = minimumTiltAngleValue
        self.maximumTiltAngleValue = maximumTiltAngleValue
        self.minimumCenterFractionToMeasureValue = minimumCenterFractionToMeasureValue
        self.maximumCenterFractionToMeasureValue = maximumCenterFractionToMeasureValue

    def stringNumberRangeValidator(self, proposedText, minimumValue, maximumValue):
        if proposedText == '':
            return True
        if not proposedText.replace('.', '', 1).isdigit():
            self.tkWindow.bell()
            return False
        numberFloat = strToFloat(proposedText)
        if minimumValue <= numberFloat <= maximumValue:
            return True
        self.tkWindow.bell()
        return False

    def scaleBarWidthMicronsValidator(self, proposedText):
        return self.stringNumberRangeValidator(proposedText, self.minimumScaleBarWidthMicronsValue, self.maximumScaleBarWidthMicronsValue)

    def tiltAngleValidator(self, proposedText):
        return self.stringNumberRangeValidator(proposedText, self.minimumTiltAngleValue, self.maximumTiltAngleValue)

    def centerFractionToMeasureValidator(self, proposedText):
        return self.stringNumberRangeValidator(proposedText, self.minimumCenterFractionToMeasureValue, self.maximumCenterFractionToMeasureValue)


class NumberValidator(object):
    def __init__(self, tkWindow, numberClassesVar, classNamesVar, validClassNamesVar):
        self.tkWindow = tkWindow
        self.numberClassesVar = numberClassesVar
        self.classNamesVar = classNamesVar
        self.validClassNamesVar = validClassNamesVar

    def NumberValidate(self, proposedText):
        if proposedText == '':
            return True
        if not proposedText.replace('.', '', 1).isdigit():
            self.tkWindow.bell()
            return False
        return True

    def ClassNumberValidate(self, proposedText):
        if self.NumberValidate(proposedText):
            if proposedText:
                if checkClassNames(self.classNamesVar.get(), strToInt(proposedText)):
                    self.validClassNamesVar.set(True)
                else:
                    self.validClassNamesVar.set(False)
            return True
        return False

    def ClassListValidate(self, proposedText):
        if self.numberClassesVar.get():
            if checkClassNames(proposedText, strToInt(self.numberClassesVar.get())):
                self.validClassNamesVar.set(True)
            else:
                self.validClassNamesVar.set(False)
        return True

def hide_AdvancedOptions(win):
    if 'showPlots_Label' in win.children:
        win.children['showPlots_Label'].destroy()
        win.children['showPlots_YesButton'].destroy()
        win.children['showPlots_NoButton'].destroy()
        win.children['showBoundingBoxPlots_Label'].destroy()
        win.children['showBoundingBoxPlots_YesButton'].destroy()
        win.children['showBoundingBoxPlots_NoButton'].destroy()
        win.children['plotPolylidar_Label'].destroy()
        win.children['plotPolylidar_YesButton'].destroy()
        win.children['plotPolylidar_NoButton'].destroy()
        win.children['parallelProcessing_Label'].destroy()
        win.children['parallelProcessing_YesButton'].destroy()
        win.children['parallelProcessing_NoButton'].destroy()


def show_AdvancedOptions(win, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar):
    if 'showPlots_Label' not in win.children:
        initialRow = 15
        tkinter.Label(win, text="Show Intermediate Plots?", name='showPlots_Label').grid(row=initialRow, column=0)
        tkinter.Radiobutton(win, text="Yes", variable=showPlotsVar, value=1, name='showPlots_YesButton').grid(row=initialRow, column=1)
        tkinter.Radiobutton(win, text="No", variable=showPlotsVar, value=0, name='showPlots_NoButton').grid(row=initialRow, column=2)

        tkinter.Label(win, text="Show BoundingBox Plots?", name='showBoundingBoxPlots_Label').grid(row=initialRow+1, column=0)
        tkinter.Radiobutton(win, text="Yes", variable=showBoundingBoxPlotsVar, value=1, name='showBoundingBoxPlots_YesButton').grid(row=initialRow+1, column=1)
        tkinter.Radiobutton(win, text="No", variable=showBoundingBoxPlotsVar, value=0, name='showBoundingBoxPlots_NoButton').grid(row=initialRow+1, column=2)

        tkinter.Label(win, text="Show Polylidar Point Plot?", name='plotPolylidar_Label').grid(row=initialRow+2, column=0)
        tkinter.Radiobutton(win, text="Yes", variable=plotPolylidarVar, value=1, name='plotPolylidar_YesButton').grid(row=initialRow+2, column=1)
        tkinter.Radiobutton(win, text="No", variable=plotPolylidarVar, value=0, name='plotPolylidar_NoButton').grid(row=initialRow+2, column=2)

        tkinter.Label(win, text="Use parallelization?", name='parallelProcessing_Label').grid(row=initialRow+3, column=0)
        tkinter.Radiobutton(win, text="Yes", variable=parallelProcessingVar, value=1, name='parallelProcessing_YesButton').grid(row=initialRow+3, column=1)
        tkinter.Radiobutton(win, text="No", variable=parallelProcessingVar, value=0, name='parallelProcessing_NoButton').grid(row=initialRow+3, column=2)

    else:
        hide_AdvancedOptions(win)


def get_file(entryField, entryFieldText, titleMessage, fileFormatsStr):
    listName = getFileOrDir('file', titleMessage, fileFormatsStr, entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def preview_image(imageFilePath):
    inputImagePath = imageFilePath.get().replace('~', os.path.expanduser('~'))
    if inputImagePath.endswith(('.tiff', '.tif')):
        print("attempting to convert tiff to png")
        rawImage = Image.open(inputImagePath)
        npImage = ((np.array(rawImage) + 1) / 256) - 1
        visImage = Image.fromarray(np.uint8(npImage), mode='L')
        visImage.show()
    else:
        rawImage = Image.open(inputImagePath)
        rawImage.show()


def get_setupOptions(savedJSONFileName):
    try:
        with open(savedJSONFileName) as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleDictEntryText, modelEntryText, isVerticalSubSectionVar, centerFractionToMeasureVar, tiltAngleVar, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar, scaleBarWidthMicronsVar, numberClassesVar, classNamesVar, wireMeasurementsText):
    setupOptions.imageFilePath = ImageEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.scaleDictPath = scaleDictEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.modelPath = modelEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.wireMeasurementsPath = wireMeasurementsText.get().replace('~', os.path.expanduser('~'))
    setupOptions.isVerticalSubSection = isVerticalSubSectionVar.get()
    setupOptions.centerFractionToMeasure = strToFloat(centerFractionToMeasureVar.get())
    setupOptions.tiltAngle = strToFloat(tiltAngleVar.get())
    setupOptions.showPlots = showPlotsVar.get()
    setupOptions.showBoundingBoxPlots = showBoundingBoxPlotsVar.get()
    setupOptions.plotPolylidar = plotPolylidarVar.get()
    setupOptions.parallelProcessing = parallelProcessingVar.get()
    setupOptions.scaleBarWidthMicrons = strToFloat(scaleBarWidthMicronsVar.get())
    setupOptions.numClasses = strToInt(numberClassesVar.get())
    setupOptions.classNameList = [entry for entry in lineSplitter(classNamesVar.get()) if entry]

    with open(savedJSONFileName, 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions, savedJSONFileName):
    win.title("ML Analysis UI")
    ImageEntryText = tkinter.StringVar(value=setupOptions.imageFilePath.replace(os.path.expanduser('~'), '~'))
    scaleDictEntryText = tkinter.StringVar(value=setupOptions.scaleDictPath.replace(os.path.expanduser('~'), '~'))
    modelEntryText = tkinter.StringVar(value=setupOptions.modelPath.replace(os.path.expanduser('~'), '~'))
    wireMeasurementsText = tkinter.StringVar(value=setupOptions.wireMeasurementsPath.replace(os.path.expanduser('~'), '~'))

    isVerticalSubSectionVar = tkinter.BooleanVar(value=setupOptions.isVerticalSubSection)
    scaleBarWidthMicronsVar = tkinter.StringVar(value=setupOptions.scaleBarWidthMicrons)
    centerFractionToMeasureVar = tkinter.StringVar(value=setupOptions.centerFractionToMeasure)
    tiltAngleVar = tkinter.StringVar(value=setupOptions.tiltAngle)
    numberClassesVar = tkinter.StringVar(value=setupOptions.numClasses)
    classNamesVar = tkinter.StringVar(value=listToCommaString(setupOptions.classNameList))
    validClassNamesVar = tkinter.BooleanVar(value=checkClassNames(classNamesVar.get(), int(numberClassesVar.get())))

    showPlotsVar = tkinter.BooleanVar(value=setupOptions.showPlots)
    showBoundingBoxPlotsVar = tkinter.BooleanVar(value=setupOptions.showBoundingBoxPlots)
    plotPolylidarVar = tkinter.BooleanVar(value=setupOptions.plotPolylidar)
    parallelProcessingVar = tkinter.BooleanVar(value=setupOptions.parallelProcessing)

    tkinter.Label(win, text="Image File:").grid(row=0, column=0)
    ImageFileEntry = tkinter.Entry(win, textvariable=ImageEntryText, width=len(setupOptions.imageFilePath.replace(os.path.expanduser('~'), '~')))
    ImageFileEntry.grid(row=1, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(ImageFileEntry, ImageEntryText, 'Choose Image File', '.jpg .jpeg .png .tiff .tif')).grid(row=1, column=1)
    tkinter.Button(win, text='Preview', command=lambda: preview_image(ImageEntryText)).grid(row=1, column=2)

    tkinter.Label(win, text="Scale Dict File:").grid(row=2, column=0)
    scaleDictEntry = tkinter.Entry(win, textvariable=scaleDictEntryText, width=len(setupOptions.scaleDictPath.replace(os.path.expanduser('~'), '~')))
    scaleDictEntry.grid(row=3, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(scaleDictEntry, scaleDictEntryText, 'Choose Scale Dict File', '.txt')).grid(row=3, column=1)

    tkinter.Label(win, text="Machine Learning Model:").grid(row=4, column=0)
    modelEntry = tkinter.Entry(win, textvariable=modelEntryText, width=len(setupOptions.modelPath.replace(os.path.expanduser('~'), '~')))
    modelEntry.grid(row=5, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(modelEntry, modelEntryText, 'Choose Machine Learning Model', '.pth')).grid(row=5, column=1)

    tkinter.Label(win, text="Line Measurement File:").grid(row=6, column=0)
    wireMeasurements = tkinter.Entry(win, textvariable=wireMeasurementsText, width=len(setupOptions.wireMeasurementsPath.replace(os.path.expanduser('~'), '~')))
    wireMeasurements.grid(row=7, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(wireMeasurements, wireMeasurementsText, 'Choose Line Measurements File', '.txt')).grid(row=7, column=1)

    tkinter.Label(win, text="Measure Widths or Lengths").grid(row=8, column=0)
    tkinter.Radiobutton(win, text="Widths", variable=isVerticalSubSectionVar, value=1).grid(row=8, column=1)
    tkinter.Radiobutton(win, text="Lengths", variable=isVerticalSubSectionVar, value=0).grid(row=8, column=2)

    txtValidator = TextValidator(win, minimumScaleBarWidthMicronsValue=0, maximumScaleBarWidthMicronsValue=1000, minimumTiltAngleValue=-10, maximumTiltAngleValue=90, minimumCenterFractionToMeasureValue=0, maximumCenterFractionToMeasureValue=1)
    scaleBarWidthMicronsValidatorFunction = (win.register(txtValidator.scaleBarWidthMicronsValidator), '%P')
    tkinter.Label(win, text="Scale Bar Size (Microns)").grid(row=9, column=0)
    tkinter.Entry(win, textvariable=scaleBarWidthMicronsVar, validate='all', validatecommand=scaleBarWidthMicronsValidatorFunction).grid(row=9, column=1)

    tiltAngleValidatorFunction = (win.register(txtValidator.tiltAngleValidator), '%P')
    tkinter.Label(win, text="Tilt Angle").grid(row=10, column=0)
    tkinter.Entry(win, textvariable=tiltAngleVar, validate='all', validatecommand=tiltAngleValidatorFunction).grid(row=10, column=1)

    centerFractionToMeasureValidatorFunction = (win.register(txtValidator.centerFractionToMeasureValidator), '%P')
    tkinter.Label(win, text="Center Fraction to Measure").grid(row=11, column=0)
    tkinter.Entry(win, textvariable=centerFractionToMeasureVar, validate='all', validatecommand=centerFractionToMeasureValidatorFunction).grid(row=11, column=1)

    numberValidator = NumberValidator(win, numberClassesVar=numberClassesVar, classNamesVar=classNamesVar, validClassNamesVar=validClassNamesVar)
    classNumberValidatorFunction = (win.register(numberValidator.ClassNumberValidate), '%P')
    classListValidatorFunction = (win.register(numberValidator.ClassListValidate), '%P')

    tkinter.Label(win, text="Number of classes in images (normally 1)").grid(row=12, column=0)
    tkinter.Entry(win, textvariable=numberClassesVar, validate='all', validatecommand=classNumberValidatorFunction).grid(row=12, column=1)

    tkinter.Label(win, text="Class names (comma separated)").grid(row=13, column=0)
    tkinter.Entry(win, textvariable=classNamesVar, validate='all', validatecommand=classListValidatorFunction).grid(row=13, column=1)

    tkinter.Button(win, text='Show/Hide Advanced Options', command=lambda: show_AdvancedOptions(win, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar)).grid(row=14, column=1)

    hide_AdvancedOptions(win)
    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleDictEntryText, modelEntryText, isVerticalSubSectionVar, centerFractionToMeasureVar, tiltAngleVar, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar, scaleBarWidthMicronsVar, numberClassesVar, classNamesVar, wireMeasurementsText))
    win.mainloop()


def setupOptionsUI():
    savedJSONFileName = 'AnalyzeOutputSetupOptions.json'
    setupOptions = get_setupOptions(savedJSONFileName)  # Read previously used setupOptions
    uiInput(Tk(), setupOptions, savedJSONFileName)
    return setupOptions


# setupOptionsUI()