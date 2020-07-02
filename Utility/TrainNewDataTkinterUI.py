import os
import re
import pickle
import json
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
from PIL import Image
from Utility.Utilities import *


def checkClassNames(classNamesString, numberClasses):
    splitLine = [entry for entry in lineSplitter(classNamesString) if entry]
    if len(splitLine) != int(numberClasses) or not classNamesString:
        warningColor = (241 / 255, 196 / 255, 15 / 255, 1)
        return False
    else:
        goodColor = (39 / 255, 174 / 255, 96 / 255, 1)
        return True


class IterationValidator(object):
    def __init__(self, tkWindow, setupOptions, modelEntryVar, folderSuffixText, completedIterationsVar):
        self.tkWindow = tkWindow
        self.modelEntryVar = modelEntryVar
        self.folderSuffixText = folderSuffixText
        self.completedIterationsVar = completedIterationsVar
        self.setupOptions = setupOptions

    def iterationValidate(self, modelType, folderSuffix):
        lastIteration = getLastIteration(outputModelFolderConverter(modelType, folderSuffix))
        if lastIteration > 0:
            lastIteration += 1
        self.completedIterationsVar.set(lastIteration)
        if lastIteration > 0:
            goodColor = (39 / 255, 174 / 255, 96 / 255, 1)
            self.setupOptions.continueTraining = True
            print("Found iterations")
        else:
            warningColor = (241 / 255, 196 / 255, 15 / 255, 1)
            self.setupOptions.continueTraining = False
            print("No found iterations")
        return True

    def folderSuffixValidate(self, proposedText):
        self.iterationValidate(self.modelEntryVar.get(), proposedText)
        return True

    def modelTypeValidate(self, selection):
        self.iterationValidate(selection, self.folderSuffixText.get())


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

    completedIterationsVar = tkinter.StringVar(value=0)
    
    totalIterationsVar = tkinter.StringVar(value=setupOptions.totalIterations)
    iterationCheckpointVar = tkinter.StringVar(value=setupOptions.iterationCheckpointPeriod)
    numberClassesVar = tkinter.StringVar(value=setupOptions.numClasses)

    classNamesVar = tkinter.StringVar(value=listToCommaString(setupOptions.classNameList))

    showPlotsVar = tkinter.StringVar(value=boolToString(setupOptions.showPlots))
    showPlotsOptions = {'True', 'False'}
    validClassNamesVar = tkinter.BooleanVar(value=checkClassNames(classNamesVar.get(), int(numberClassesVar.get())))

    tkinter.Label(win, text="Training Annotation Dictionary:").grid(row=0, column=0)
    trainingDictEntry = tkinter.Entry(win, textvariable=trainDictText, width=len(setupOptions.trainDictPath.replace(os.path.expanduser('~'), '~')))
    trainingDictEntry.grid(row=1, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(trainingDictEntry, trainDictText, 'Choose Training Annotation Dictionary', '.txt')).grid(row=1, column=1)

    tkinter.Label(win, text="Validation Annotation Dictionary:").grid(row=2, column=0)
    validationDictEntry = tkinter.Entry(win, textvariable=validationDictText, width=len(setupOptions.validationDictPath.replace(os.path.expanduser('~'), '~')))
    validationDictEntry.grid(row=3, column=0)
    tkinter.Button(win, text='Choose File', command=lambda: get_file(validationDictEntry, validationDictText, 'Choose Validation Annotation Dictionary', '.txt')).grid(row=3, column=1)

    iterationValidator = IterationValidator(win, setupOptions=setupOptions, modelEntryVar=modelEntryVar, folderSuffixText=folderSuffixText, completedIterationsVar=completedIterationsVar)
    folderSuffixValidatorFunction = (win.register(iterationValidator.folderSuffixValidate), '%P')

    tkinter.Label(win, text="Machine Learning Model:").grid(row=4, column=0)
    tkinter.OptionMenu(win, modelEntryVar, *modelEntryOptions, command=iterationValidator.modelTypeValidate).grid(row=4, column=1)

    # TODO: figure out changing text color to indicate validation/warning for if no iterations performed yet

    tkinter.Label(win, text="Model output folder name suffix:").grid(row=5, column=0)
    tkinter.Entry(win, textvariable=folderSuffixText, validate='all', validatecommand=folderSuffixValidatorFunction).grid(row=5, column=1)

    tkinter.Label(win, text="Completed iterations on chosen model:").grid(row=6, column=0)
    tkinter.Entry(win, textvariable=completedIterationsVar, width=len(completedIterationsVar.get()), state='readonly').grid(row=6, column=1)

    numberValidator = NumberValidator(win, numberClassesVar=numberClassesVar, classNamesVar=classNamesVar, validClassNamesVar=validClassNamesVar)
    numberValidatorFunction = (win.register(numberValidator.NumberValidate), '%P')
    classNumberValidatorFunction = (win.register(numberValidator.ClassNumberValidate), '%P')
    classListValidatorFunction = (win.register(numberValidator.ClassListValidate), '%P')

    tkinter.Label(win, text="Total Number of Iterations:").grid(row=7, column=0)
    tkinter.Entry(win, textvariable=totalIterationsVar, validate='all', validatecommand=numberValidatorFunction).grid(row=7, column=1)

    tkinter.Label(win, text="Model iteration checkpoint period (Number of iterations)").grid(row=8, column=0)
    tkinter.Entry(win, textvariable=iterationCheckpointVar, validate='all', validatecommand=numberValidatorFunction).grid(row=8, column=1)

    tkinter.Label(win, text="Number of classes in images (normally 1)").grid(row=9, column=0)
    tkinter.Entry(win, textvariable=numberClassesVar, validate='all', validatecommand=classNumberValidatorFunction).grid(row=9, column=1)

    tkinter.Label(win, text="Class names (comma separated)").grid(row=10, column=0)
    tkinter.Entry(win, textvariable=classNamesVar, validate='all', validatecommand=classListValidatorFunction).grid(row=10, column=1)

    tkinter.Label(win, text="Show plots with annotated images before training?").grid(row=11, column=0)
    tkinter.OptionMenu(win, showPlotsVar, *showPlotsOptions).grid(row=11, column=1)

    tkinter.Label(win, text="Valid class names list:").grid(row=12, column=0)
    tkinter.Entry(win, textvariable=validClassNamesVar, state='readonly').grid(row=12, column=1)

    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, savedJSONFileName, trainDictText, validationDictText, modelEntryVar, folderSuffixText, totalIterationsVar, iterationCheckpointVar, numberClassesVar, classNamesVar, showPlotsVar))
    win.mainloop()


def setupOptionsUI():
    savedJSONFileName = 'TrainNewDataSetupOptions.json'
    setupOptions = get_setupOptions(savedJSONFileName)  # Read previously used setupOptions
    uiInput(Tk(), setupOptions, savedJSONFileName)
    return setupOptions

