import os
import re
import pickle
import json
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
from PIL import Image
from Utility.Utilities import *


class TextValidator(object):
    def __init__(self, tkWindow, numberClassesVar, modelTypeVar, folderSuffixText):
        self.tkWindow = tkWindow
        self.numberClassesVar = numberClassesVar
        self.modelTypeVar = modelTypeVar
        self.folderSuffixText = folderSuffixText

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

    def classNameListValidator(self, proposedText):
        return self.stringNumberRangeValidator(proposedText, self.minimumScaleBarWidthMicronsValue, self.maximumScaleBarWidthMicronsValue)


class NumberValidator(object):
    def __init__(self, tkWindow):
        self.tkWindow = tkWindow

    def NumberValidate(self, proposedText):
        if proposedText == '':
            return True
        if not proposedText.replace('.', '', 1).isdigit():
            self.tkWindow.bell()
            return False
        return True


def get_file(entryField, entryFieldText, titleMessage, fileFormatsStr):
    listName = getFileOrDir('file', titleMessage, fileFormatsStr, entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def get_setupOptions(savedJSONFileName):
    try:
        with open(savedJSONFileName) as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def boolToString(boolValue):
    if boolValue:
        return "True"
    return "False"


def on_closing(win, setupOptions, savedJSONFileName, trainDictText, validationDictText, modelEntryVar, folderSuffixText, totalIterationsVar, iterationCheckpointVar, numberClassesVar, classNamesVar, showPlotsVar):
    setupOptions.trainDictPath = trainDictText.get().replace('~', os.path.expanduser('~'))
    setupOptions.validationDictPath = validationDictText.get().replace('~', os.path.expanduser('~'))
    setupOptions.modelType = modelEntryVar.get()
    setupOptions.folderSuffix = folderSuffixText.get()
    setupOptions.totalIterations = strToInt(totalIterationsVar.get())
    setupOptions.iterationCheckpointPeriod = strToInt(iterationCheckpointVar.get())
    setupOptions.numClasses = strToInt(numberClassesVar.get())
    setupOptions.classNameList = [entry for entry in lineSplitter(classNamesVar.get()) if entry]
    setupOptions.showPlots = textToBool(showPlotsVar.get())

    with open(savedJSONFileName, 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions, savedJSONFileName):
    win.title("ML Training UI")
    trainDictText = tkinter.StringVar(value=setupOptions.trainDictPath.replace(os.path.expanduser('~'), '~'))
    validationDictText = tkinter.StringVar(value=setupOptions.validationDictPath.replace(os.path.expanduser('~'), '~'))
    modelEntryVar = tkinter.StringVar(value=setupOptions.modelType.replace(os.path.expanduser('~'), '~'))
    modelEntryOptions = {'MaskRCNN', 'PointRend'}
    folderSuffixText = tkinter.StringVar(value=setupOptions.folderSuffix)

    # completedIterationsVar = tkinter.StringVar(value=0)
    
    totalIterationsVar = tkinter.StringVar(value=setupOptions.totalIterations)
    iterationCheckpointVar = tkinter.StringVar(value=setupOptions.iterationCheckpointPeriod)
    numberClassesVar = tkinter.StringVar(value=setupOptions.numClasses)

    classNamesVar = tkinter.StringVar(value=listToCommaString(setupOptions.classNameList))

    showPlotsVar = tkinter.StringVar(value=boolToString(setupOptions.showPlots))
    showPlotsOptions = {'True', 'False'}

    tkinter.Label(win, text="Training Annotation Dictionary:").grid(row=0, column=0)
    trainingDictEntry = tkinter.Entry(win, textvariable=trainDictText, width=len(setupOptions.trainDictPath.replace(os.path.expanduser('~'), '~')))
    trainingDictEntry.grid(row=1, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(trainingDictEntry, trainDictText, 'Choose Training Annotation Dictionary', '.txt')).grid(row=1, column=1)

    tkinter.Label(win, text="Validation Annotation Dictionary:").grid(row=2, column=0)
    validationDictEntry = tkinter.Entry(win, textvariable=validationDictText, width=len(setupOptions.validationDictPath.replace(os.path.expanduser('~'), '~')))
    validationDictEntry.grid(row=3, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(validationDictEntry, validationDictText, 'Choose Validation Annotation Dictionary', '.txt')).grid(row=3, column=1)

    tkinter.Label(win, text="Machine Learning Model:").grid(row=4, column=0)
    tkinter.OptionMenu(win, modelEntryVar, *modelEntryOptions).grid(row=4, column=1)

    # TODO: figure out validation, possibly changing text color
    # txtValidator = TextValidator(win, numberClasses=numberClassesVar, modelType=modelEntryVar, folderSuffix=folderSuffixText)
    tkinter.Label(win, text="Model output folder name suffix:").grid(row=5, column=0)
    tkinter.Entry(win, textvariable=folderSuffixText).grid(row=5, column=1)
    # tkinter.Entry(win, textvariable=folderSuffixText, width=len(setupOptions.folderSuffix), validate='all', validatecommand=scaleBarWidthMicronsValidatorFunction).grid(row=5, column=1)
    #
    # tkinter.Label(win, text="Completed iterations on chosen model:").grid(row=6, column=0)
    # tkinter.Entry(win, textvariable=completedIterationsVar, width=len(completedIterationsVar.get()), state='readonly').grid(row=6, column=1)

    numberValidator = NumberValidator(win)
    numberValidatorFunction = (win.register(numberValidator.NumberValidate), '%P')

    tkinter.Label(win, text="Total Number of Iterations:").grid(row=7, column=0)
    tkinter.Entry(win, textvariable=totalIterationsVar, validate='all', validatecommand=numberValidatorFunction).grid(row=7, column=1)

    tkinter.Label(win, text="Model iteration checkpoint period (Number of iterations)").grid(row=8, column=0)
    tkinter.Entry(win, textvariable=iterationCheckpointVar, validate='all', validatecommand=numberValidatorFunction).grid(row=8, column=1)

    tkinter.Label(win, text="Number of classes in images (normally 1)").grid(row=9, column=0)
    tkinter.Entry(win, textvariable=numberClassesVar, validate='all', validatecommand=numberValidatorFunction).grid(row=9, column=1)

    tkinter.Label(win, text="Class names (comma separated)").grid(row=10, column=0)
    tkinter.Entry(win, textvariable=classNamesVar).grid(row=10, column=1)

    tkinter.Label(win, text="Show plots with annotated images before training?").grid(row=11, column=0)
    tkinter.OptionMenu(win, showPlotsVar, *showPlotsOptions).grid(row=11, column=1)

    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, savedJSONFileName, trainDictText, validationDictText, modelEntryVar, folderSuffixText, totalIterationsVar, iterationCheckpointVar, numberClassesVar, classNamesVar, showPlotsVar))
    win.mainloop()


def setupOptionsUI():
    savedJSONFileName = 'TrainNewDataSetupOptions.json'
    setupOptions = get_setupOptions(savedJSONFileName)  # Read previously used setupOptions
    uiInput(Tk(), setupOptions, savedJSONFileName)
    return setupOptions

