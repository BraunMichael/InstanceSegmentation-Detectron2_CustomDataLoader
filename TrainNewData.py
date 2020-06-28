import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import multiprocessing
import random
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

from torch import load as torchload
from torch import device as torchdevice
from glob import glob

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2_repo.projects.PointRend import point_rend
from detectron2.utils.logger import setup_logger
setup_logger()

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
    def __init__(self, **kwargs):
        self.title = "Deep Learning Training Setup UI"
        self.theme_cls.primary_palette = "Blue"
        super().__init__(**kwargs)
        self.root = Factory.SetupUI()
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
            setupoptions.continueTraining = True
        else:
            warningColor = (241/255, 196/255, 15/255, 1)
            self.root.ids['iterationsComplete'].line_color_focus = warningColor
            self.root.ids['iterationsComplete'].text = str(lastIteration)
            self.root.ids['iterationsCompleteLabel'].text_color = warningColor
            self.root.ids['iterationsCompleteLabel'].text = "Warning, there are no detected iterations on chosen model path. Will start from pre-trained model only."
            setupoptions.continueTraining = False

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
                setupoptions.showPlots = textToBool(entry.current_item)
            elif key == 'numClassesField':
                setupoptions.numClasses = int(entry.text)
            elif key == 'folderSuffixField':
                setupoptions.folderSuffix = entry.text
            elif key == 'modelTypeButton':
                setupoptions.modelType = entry.current_item
            elif key == 'iterationCheckpointPeriod':
                charFreeStr = ''.join(ch for ch in entry.text if ch.isdigit() or ch == '.' or ch == ',')
                setupoptions.iterationCheckpointPeriod = int(float(locale.atof(charFreeStr)))
            elif key == 'totalIterations':
                charFreeStr = ''.join(ch for ch in entry.text if ch.isdigit() or ch == '.' or ch == ',')
                setupoptions.totalIterations = int(float(locale.atof(charFreeStr)))
            elif key == 'trainAnnotationDictPath':
                setupoptions.trainDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'validateAnnotationDictPath':
                setupoptions.validationDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'classNamesField':
                setupoptions.classNameList = lineSplitter(entry.text)

        self.root_window.close()


def setConfigurator(setupoptions: SetupOptions, baseStr: str = '', maskType: str = ''):
    modelType = setupoptions.modelType
    outputModelFolder = outputModelFolderConverter(modelType, setupoptions.folderSuffix)
    numClasses = setupoptions.numClasses  # only has one class (VerticalNanowires)

    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(os.getcwd(), 'OutputModels', outputModelFolder)

    if modelType.lower() == 'pointrend':
        print('PointRend Model')
        # See params here https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend/point_rend/config.py
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(os.path.join(os.getcwd(), "detectron2_repo", "projects", "PointRend", "configs", "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = numClasses  # PointRend has to match num classes
        if setupoptions.continueTraining:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        else:
            cfg.MODEL.WEIGHTS = os.path.join('https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl')
    elif modelType.lower() == 'maskrcnn':
        print('MaskRCNN Model')
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        if setupoptions.continueTraining:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = numClasses
    cfg.INPUT.MASK_FORMAT = maskType.lower()
    cfg.DATASETS.TRAIN = (baseStr + "_Train",)
    cfg.DATASETS.TEST = (baseStr + "_Validation",)
    cfg.DATALOADER.NUM_WORKERS = multiprocessing.cpu_count()
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = setupoptions.totalIterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512, balloon test used 128)

    cfg.INPUT.MIN_SIZE_TRAIN = (1179,)  # (default: (800,))
    cfg.INPUT.MAX_SIZE_TRAIN = 1366  # (default: 1333)
    cfg.TEST.DETECTIONS_PER_IMAGE = 200  # Increased from COCO default, should never have more than 200 wires per image (default: 100)
    cfg.SOLVER.CHECKPOINT_PERIOD = setupoptions.iterationCheckpointPeriod
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000  # (default: 12000)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # (default: 6000)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def setDatasetAndMetadata(baseStr: str, setupoptions: SetupOptions):
    showPlots = setupoptions.showPlots

    annotationTrainListFileName = setupoptions.trainDictPath
    annotationValidateListFileName = setupoptions.validationDictPath

    # Need to make a train and a validation list of dicts separately in InstanceSegmentationDatasetDict
    annotationTrainDicts = fileHandling(annotationTrainListFileName)
    annotationValidateDicts = fileHandling(annotationValidateListFileName)

    annotationDicts = [annotationTrainDicts, annotationValidateDicts]

    # Just loop through each dict and get the file name, split the path to get the parent dir then check if those are Train and Validation
    dirNameSet = set()
    maskTypeSet = set()
    for annotationDictList in annotationDicts:
        for annotationDict in annotationDictList:
            parentDirName = os.path.split(os.path.split(annotationDict['file_name'])[0])[-1]
            if parentDirName not in dirNameSet:
                dirNameSet.add(parentDirName)
            if isinstance(annotationDict['annotations'][0], list):
                fileMaskType = 'polygon'
            elif isinstance(annotationDict['annotations'][0], dict):
                fileMaskType = 'bitmask'
            else:
                fileMaskType = ''
            assert fileMaskType, 'The annotation dict annotations did not match the expected pattern for polygon or bitmask encoding. Check your annotation creation.'
            if fileMaskType not in maskTypeSet:
                maskTypeSet.add(fileMaskType)

    # dirNameSet should return {'Train', 'Validation'}
    assert len(maskTypeSet) == 1, "The number of detected mask types is not 1, check your annotation creation and file choice."
    maskType = list(maskTypeSet)[0]  # There is only 1 entry, assert checks that above

    assert 'Train' in dirNameSet and 'Validation' in dirNameSet, 'You are missing either a Train or Validation directory in your annotations'
    dirnames = ['Train', 'Validation']  # After making sure these are directories as expected, lets force the order to match the annotationDicts order

    for d in range(len(dirnames)):
        if baseStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
            DatasetCatalog.register(baseStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
        MetadataCatalog.get(baseStr + "_" + dirnames[d]).set(thing_classes=setupoptions.classNameList)

    if showPlots:
        nanowire_metadata = MetadataCatalog.get(baseStr + "_Train")
        for d in random.sample(annotationTrainDicts, 5):
            fig, ax = plt.subplots(figsize=(10, 8))
            print(d["file_name"])
            rawImage = Image.open(d["file_name"])
            npImage = np.array(rawImage)
            # TODO fix this
            try:
                visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
            except IndexError:
                npImage = np.expand_dims(npImage, axis=2)
                visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
            visTest = visualizerNP.draw_dataset_dict(d)
            ax.imshow(visTest.get_image()[:, :, ::-1])
            plt.show(block=True)
    return maskType


def textToBool(text):
    assert text.lower() == 'true' or text.lower() == 'false', "The passed text is not true/false"
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False


def main(setupoptions: SetupOptions):
    baseStr = 'VerticalNanowires'

    maskType = setDatasetAndMetadata(baseStr, setupoptions)
    configurator = setConfigurator(setupoptions, baseStr, maskType)
    trainer = DefaultTrainer(configurator)
    trainer.resume_or_load(resume=setupoptions.continueTraining)
    trainer.train()


if __name__ == "__main__":
    setupoptions = SetupOptions()
    Window.size = (750, 670)
    Builder.load_file(f"TrainNewDataUI.kv")
    SetupUI().run()
    main(setupoptions)