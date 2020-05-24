import os
import json
import jsonpickle
import tkinter
from tkinter import Tk, filedialog
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class SetupOptions:
    def __init__(self):
        self.imageFilePath = ''
        self.scaleDictPath = ''
        self.isVerticalSubSection = True
        self.centerFractionToMeasure = 0.5
        self.tiltAngle = 30
        self.showPlots = False
        self.showBoundingBoxPlots = False
        self.plotPolylidar = False
        self.parallelProcessing = True


def strToFloat(numberString):
    charFreeStr = ''.join(ch for ch in numberString if ch.isdigit() or ch == '.' or ch == ',')
    return float(locale.atof(charFreeStr))


def validStringNumberRange(numberString, minimumValue, maximumValue):
    if minimumValue < strToFloat(numberString) < maximumValue:
        return True
    return False


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
        item_Label = tkinter.Label(win, text="Show Intermediate Plots?", name='showPlots_Label')
        item_Label.grid(row=8, column=0)
        r1showPlots = tkinter.Radiobutton(win, text="Yes", variable=showPlotsVar, value=1, name='showPlots_YesButton')
        r2showPlots = tkinter.Radiobutton(win, text="No", variable=showPlotsVar, value=0, name='showPlots_NoButton')
        r1showPlots.grid(row=8, column=1)
        r2showPlots.grid(row=8, column=2)
        item_Label = tkinter.Label(win, text="Show BoundingBox Plots?", name='showBoundingBoxPlots_Label')
        item_Label.grid(row=9, column=0)
        r1showBoundingBoxPlots = tkinter.Radiobutton(win, text="Yes", variable=showBoundingBoxPlotsVar, value=1, name='showBoundingBoxPlots_YesButton')
        r2showBoundingBoxPlots = tkinter.Radiobutton(win, text="No", variable=showBoundingBoxPlotsVar, value=0, name='showBoundingBoxPlots_NoButton')
        r1showBoundingBoxPlots.grid(row=9, column=1)
        r2showBoundingBoxPlots.grid(row=9, column=2)
        item_Label = tkinter.Label(win, text="Show Polylidar Point Plot?", name='plotPolylidar_Label')
        item_Label.grid(row=10, column=0)
        r1plotPolylidar = tkinter.Radiobutton(win, text="Yes", variable=plotPolylidarVar, value=1, name='plotPolylidar_YesButton')
        r2plotPolylidar = tkinter.Radiobutton(win, text="No", variable=plotPolylidarVar, value=0, name='plotPolylidar_NoButton')
        r1plotPolylidar.grid(row=10, column=1)
        r2plotPolylidar.grid(row=10, column=2)
        item_Label = tkinter.Label(win, text="Use parallelization?", name='parallelProcessing_Label')
        item_Label.grid(row=11, column=0)
        r1parallelProcessing = tkinter.Radiobutton(win, text="Yes", variable=parallelProcessingVar, value=1, name='parallelProcessing_YesButton')
        r2parallelProcessing = tkinter.Radiobutton(win, text="No", variable=parallelProcessingVar, value=0, name='parallelProcessing_NoButton')
        r1parallelProcessing.grid(row=11, column=1)
        r2parallelProcessing.grid(row=11, column=2)
    else:
        hide_AdvancedOptions(win)


def getFileOrDirList(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = None,
                     initialDirOrFile: str = os.getcwd()):
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


def get_file(entryField, entryFieldText, titleMessage, fileFormatsStr):
    listName = getFileOrDirList('file', titleMessage, fileFormatsStr, entryFieldText.get().replace('~', os.path.expanduser('~')))
    entryFieldText.set(listName.replace(os.path.expanduser('~'), '~'))
    entryField.config(width=len(listName.replace(os.path.expanduser('~'), '~')))


def preview_image(imageFilePath):
    print('not implemented yet')


def get_setupOptions(savedJSONFileName):
    try:
        with open(savedJSONFileName) as infile:
            inputFile = json.load(infile)
        setupOptions = jsonpickle.decode(inputFile)
    except FileNotFoundError:
        setupOptions = SetupOptions()
    return setupOptions


def on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleDictEntryText, isVerticalSubSectionVar, centerFractionToMeasureVar, tiltAngleVar, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar):
    setupOptions.imageFilePath = ImageEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.scaleDictPath = scaleDictEntryText.get().replace('~', os.path.expanduser('~'))
    setupOptions.isVerticalSubSection = isVerticalSubSectionVar.get()
    setupOptions.centerFractionToMeasure = strToFloat(centerFractionToMeasureVar.get())
    setupOptions.tiltAngle = strToFloat(tiltAngleVar.get())
    setupOptions.showPlots = showPlotsVar.get()
    setupOptions.showBoundingBoxPlots = showBoundingBoxPlotsVar.get()
    setupOptions.plotPolylidar = plotPolylidarVar.get()
    setupOptions.parallelProcessing = parallelProcessingVar.get()

    with open(savedJSONFileName, 'w') as outfile:
        json.dump(jsonpickle.encode(setupOptions), outfile)
    win.destroy()


def uiInput(win, setupOptions, savedJSONFileName):
    win.title("Spectrum Data Processing Setup UI")
    ImageEntryText = tkinter.StringVar(value=setupOptions.imageFilePath.replace(os.path.expanduser('~'), '~'))
    scaleDictEntryText = tkinter.StringVar(value=setupOptions.scaleDictPath.replace(os.path.expanduser('~'), '~'))

    isVerticalSubSectionVar = tkinter.BooleanVar(value=setupOptions.isVerticalSubSection)
    centerFractionToMeasureVar = tkinter.StringVar(value=setupOptions.centerFractionToMeasure)
    tiltAngleVar = tkinter.StringVar(value=setupOptions.tiltAngle)

    showPlotsVar = tkinter.BooleanVar(value=setupOptions.showPlots)
    showBoundingBoxPlotsVar = tkinter.BooleanVar(value=setupOptions.showBoundingBoxPlots)
    plotPolylidarVar = tkinter.BooleanVar(value=setupOptions.plotPolylidar)
    parallelProcessingVar = tkinter.BooleanVar(value=setupOptions.parallelProcessing)

    tkinter.Label(win, text="Image File:").grid(row=0, column=0)
    ImageFileEntry = tkinter.Entry(win, textvariable=ImageEntryText)
    ImageFileEntry.grid(row=1, column=0)
    ImageFileEntry.config(width=len(setupOptions.imageFilePath.replace(os.path.expanduser('~'), '~')))
    ImageFileButton = tkinter.Button(win, text='Choose File', command=lambda: get_file(ImageFileEntry, ImageEntryText, 'Choose Image File', '.jpg .jpeg .png .tiff'))
    ImageFileButton.grid(row=1, column=1)
    ImageFilePreviewButton = tkinter.Button(win, text='Preview', command=lambda: preview_image(ImageEntryText))
    ImageFilePreviewButton.grid(row=1, column=2)

    tkinter.Label(win, text="Scale Dict File:").grid(row=2, column=0)
    scaleDictEntry = tkinter.Entry(win, textvariable=scaleDictEntryText)
    scaleDictEntry.grid(row=3, column=0)
    scaleDictEntry.config(width=len(setupOptions.scaleDictPath.replace(os.path.expanduser('~'), '~')))
    scaleDictButton = tkinter.Button(win, text='Choose File', command=lambda: get_file(scaleDictEntry, scaleDictEntryText, 'Choose Scale Dict File', '.txt'))
    scaleDictButton.grid(row=3, column=1)

    item_Label = tkinter.Label(win, text="Measure Widths or Lengths")
    item_Label.grid(row=4, column=0)
    r1isXRD = tkinter.Radiobutton(win, text="Widths", variable=isVerticalSubSectionVar, value=1)
    r2isXRD = tkinter.Radiobutton(win, text="Lengths", variable=isVerticalSubSectionVar, value=0)
    r1isXRD.grid(row=4, column=1)
    r2isXRD.grid(row=4, column=2)

    tkinter.Label(win, text="Tilt Angle").grid(row=5, column=0)
    tiltAngleEntry = tkinter.Entry(win, textvariable=tiltAngleVar, validate='all', validatecommand=lambda: validStringNumberRange(tiltAngleVar.get(), -10, 90))
    tiltAngleEntry.grid(row=5, column=1)

    tkinter.Label(win, text="Center Fraction to Measure").grid(row=6, column=0)
    centerFractionToMeasureEntry = tkinter.Entry(win, textvariable=centerFractionToMeasureVar, validate='all', validatecommand=lambda: validStringNumberRange(centerFractionToMeasureVar.get(), 0, 1))
    centerFractionToMeasureEntry.grid(row=6, column=1)

    tkinter.Button(win, text='Show/Hide Advanced Options', command=lambda: show_AdvancedOptions(win, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar)).grid(row=7, column=1)

    hide_AdvancedOptions(win)
    # TODO: edit lambda here, and the on_closing probably needs to convert strings to floats for tilt and centerfractiontomeasure
    win.protocol("WM_DELETE_WINDOW", lambda: on_closing(win, setupOptions, savedJSONFileName, ImageEntryText, scaleDictEntryText, isVerticalSubSectionVar, centerFractionToMeasureVar, tiltAngleVar, showPlotsVar, showBoundingBoxPlotsVar, plotPolylidarVar, parallelProcessingVar))
    win.mainloop()


def getImageAndScale():
    savedJSONFileName = 'AnalyzeOutputSetupOptions.json'
    setupOptions = get_setupOptions(savedJSONFileName)  # Read previously used setupOptions
    uiInput(Tk(), setupOptions, savedJSONFileName)


getImageAndScale()