import os
import re
import pickle
from PIL import Image
import locale
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import NumericProperty
from kivymd.app import MDApp
from kivy.storage.jsonstore import JsonStore
from kivymd.uix.textfield import MDTextField
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dropdownitem import MDDropDownItem
from kivy.core.window import Window
from Utility.Utilities import *
from glob import glob
from torch import load as torchload
from torch import device as torchdevice
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def outputModelFolderConverter(prefix: str, suffix: str):
    return prefix + "Model_" + suffix


def lineSplitter(lineString):
    delimiters = ' ', ', ', ',', '\t', '\n'
    regexPattern = '|'.join(map(re.escape, delimiters))
    splitLineList = re.split(regexPattern, lineString)
    return splitLineList


class CenteredMDTextField(MDTextField):
    text_width = NumericProperty()

    def update_padding(self):
        # Update the padding so the text is centered
        if len(self.text) > 0:
            charFreeStr = ''.join(ch for ch in self.text if ch.isdigit() or ch == '.' or ch == ',')
            self.text = format(int(float(locale.atof(charFreeStr))), ",.0f")
        self.text_width = self._get_text_width(
            self.text,
            self.tab_width,
            self._label_cached
        )


def textToBool(text):
    assert text.lower() == 'true' or text.lower() == 'false', "The passed text is not true/false"
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False


def getLastIteration(saveDir) -> int:
    """
    Returns:
        int : Number of iterations performed from model in target directory.
    """
    fullSaveDir = os.path.join(os.getcwd(), 'OutputModels', saveDir)
    checkpointFilePath = os.path.join(os.getcwd(), 'OutputModels', fullSaveDir, "last_checkpoint")

    # get file from checkpointFilePath as latestModel
    if os.path.exists(checkpointFilePath):
        with open(checkpointFilePath) as f:
            latestModel = os.path.join(fullSaveDir, f.read().strip())
    elif os.path.exists(os.path.join(fullSaveDir, 'model_final.pth')):
        latestModel = os.path.join(fullSaveDir, 'model_final.pth')
    else:
        fileList = glob("*.pth")
        if fileList:
            latestModel = sorted(fileList, reverse=True)[0]
        else:
            return 0

    latestIteration = torchload(latestModel, map_location=torchdevice("cpu")).get("iteration", -1)
    return latestIteration


class SetupUI(MDApp):
    def __init__(self, setupoptions, **kwargs):
        self.title = "Deep Learning Training Setup UI"
        self.theme_cls.primary_palette = "Blue"
        super().__init__(**kwargs)
        self.root = Factory.SetupUI()
        self.setupoptions = setupoptions
        modelTypeMenu_items = [{"text": "MaskRCNN"}, {"text": "PointRend"}]

        self.modelTypeMenu = MDDropdownMenu(
            caller=self.root.ids['modelTypeButton'],
            items=modelTypeMenu_items,
            position="auto",
            callback=self.set_model,
            width_mult=4,
        )

        trueFalseMenu_items = [{"text": "True"}, {"text": "False"}]
        self.showPlotsMenu = MDDropdownMenu(
            caller=self.root.ids['showPlotsButton'],
            items=trueFalseMenu_items,
            position="auto",
            callback=self.set_showPlots,
            width_mult=4,
        )

    def file_manager_open(self, destinationField, openType, message):
        listName = getFileOrDir(openType, message, '.txt', self.root.ids[destinationField].text.replace('~', os.path.expanduser('~')))
        self.root.ids[destinationField].text = listName.replace(os.path.expanduser('~'), '~')

    def set_model(self, instance):
        self.root.ids['modelTypeButton'].set_item(instance.text)
        self.root.ids['folderSuffixField'].helper_text = "Folder prefix currently is: " + outputModelFolderConverter(instance.text, '')
        self.setLastIteration(self.setOutputModelFolderConverter(instance.text, self.root.ids['folderSuffixField'].text))
        self.modelTypeMenu.dismiss()

    def set_showPlots(self, instance):
        self.root.ids['showPlotsButton'].set_item(instance.text)
        self.showPlotsMenu.dismiss()

    def build(self):
        return self.root

    @staticmethod
    def setOutputModelFolderConverter(prefix: str, suffix: str):
        return outputModelFolderConverter(prefix, suffix)

    def setLastIteration(self, modelDir):
        lastIteration = getLastIteration(modelDir)
        self.root.ids['iterationsComplete'].color_mode = "custom"
        self.root.ids['iterationsCompleteLabel'].theme_text_color = "Custom"

        if lastIteration > 0:
            goodColor = (39/255, 174/255, 96/255, 1)
            self.root.ids['iterationsComplete'].line_color_focus = goodColor
            self.root.ids['iterationsComplete'].text = str(lastIteration + 1)
            self.root.ids['iterationsCompleteLabel'].text_color = 0, 0, 0, 1
            self.root.ids['iterationsCompleteLabel'].text = "Completed iterations on chosen model"
            self.setupoptions.continueTraining = True
        else:
            warningColor = (241/255, 196/255, 15/255, 1)
            self.root.ids['iterationsComplete'].line_color_focus = warningColor
            self.root.ids['iterationsComplete'].text = str(lastIteration)
            self.root.ids['iterationsCompleteLabel'].text_color = warningColor
            self.root.ids['iterationsCompleteLabel'].text = "Warning, there are no detected iterations on chosen model path. Will start from pre-trained model only."
            self.setupoptions.continueTraining = False

    def checkClassNames(self, classNamesString):
        self.root.ids['iterationsComplete'].color_mode = "custom"
        self.root.ids['iterationsCompleteLabel'].theme_text_color = "Custom"

        splitLine = [entry for entry in lineSplitter(classNamesString) if entry]
        if len(splitLine) != int(self.root.ids['numClassesField'].text) or not classNamesString:
            warningColor = (241 / 255, 196 / 255, 15 / 255, 1)
            self.root.ids['classNamesField'].line_color_focus = warningColor
            self.root.ids['classNamesField'].helper_text = "The number of listed classes and stated number of classes above do not match"
        else:
            goodColor = (39 / 255, 174 / 255, 96 / 255, 1)
            self.root.ids['classNamesField'].line_color_focus = goodColor
            self.root.ids['classNamesField'].helper_text = ""
        print(splitLine)
        print(self.root.ids['classNamesField'].text)

    def on_start(self):
        self.root.ids['fileManager_Train'].ids['lbl_txt'].halign = 'center'
        self.root.ids['fileManager_Validate'].ids['lbl_txt'].halign = 'center'

        store = JsonStore('SavedSetupOptions.json')
        if store.count() > 0:
            for key in store:
                if key in self.root.ids:
                    entry = self.root.ids[key]
                    if isinstance(entry, MDTextField):
                        if key == 'validateAnnotationDictPath' or key == 'trainAnnotationDictPath':
                            entry.text = store.get(key)['text'].replace(os.path.expanduser('~'), '~')
                        else:
                            entry.text = store.get(key)['text']
                    elif isinstance(entry, MDDropDownItem):
                        entry.set_item(store.get(key)['text'])
        modelDir = outputModelFolderConverter(self.root.ids['modelTypeButton'].current_item, self.root.ids['folderSuffixField'].text)
        self.setLastIteration(modelDir)
        self.root.ids['folderSuffixField'].helper_text = "Folder prefix currently is: " + outputModelFolderConverter(self.root.ids['modelTypeButton'].current_item, '')

    def on_stop(self):
        store = JsonStore('SavedSetupOptions.json')
        for key in self.root.ids:
            entry = self.root.ids[key]
            if isinstance(entry, MDTextField):
                if key == 'validateAnnotationDictPath' or key == 'trainAnnotationDictPath':
                    store.put(key, text=entry.text.replace('~', os.path.expanduser('~')))
                else:
                    store.put(key, text=entry.text)
            elif isinstance(entry, MDDropDownItem):
                store.put(key, text=entry.current_item)

            if key == 'showPlotsButton':
                self.setupoptions.showPlots = textToBool(entry.current_item)
            elif key == 'numClassesField':
                self.setupoptions.numClasses = int(entry.text)
            elif key == 'folderSuffixField':
                self.setupoptions.folderSuffix = entry.text
            elif key == 'modelTypeButton':
                self.setupoptions.modelType = entry.current_item
            elif key == 'iterationCheckpointPeriod':
                charFreeStr = ''.join(ch for ch in entry.text if ch.isdigit() or ch == '.' or ch == ',')
                self.setupoptions.iterationCheckpointPeriod = int(float(locale.atof(charFreeStr)))
            elif key == 'totalIterations':
                charFreeStr = ''.join(ch for ch in entry.text if ch.isdigit() or ch == '.' or ch == ',')
                self.setupoptions.totalIterations = int(float(locale.atof(charFreeStr)))
            elif key == 'trainAnnotationDictPath':
                self.setupoptions.trainDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'validateAnnotationDictPath':
                self.setupoptions.validationDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'classNamesField':
                self.setupoptions.classNameList = lineSplitter(entry.text)

        self.root_window.close()


def KivySetupOptionsUI():
    setupoptions = SetupOptions()
    Window.size = (750, 670)
    Builder.load_file(f"TrainNewDataUI.kv")
    SetupUI(setupoptions).run()
    return setupoptions