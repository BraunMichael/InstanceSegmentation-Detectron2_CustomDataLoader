import os
import pickle
import matplotlib.pyplot as plt
import numpy as np  # (pip install numpy)
from PIL import Image, ImageOps
from tkinter import Tk, filedialog
import multiprocessing
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import locale
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import NumericProperty
from kivy.uix.textinput import TextInput
from kivymd.app import MDApp
from kivymd.uix.textfield import MDTextField
from kivy.storage.jsonstore import JsonStore
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDRoundFlatIconButton
from kivy.core.window import Window

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.utils.logger import setup_logger
setup_logger()

# import PointRend project
import sys
sys.path.insert(1, os.path.join(os.getcwd(),"detectron2_repo", "projects", "PointRend"))
import point_rend

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class CenteredMDTextField(MDTextField):
    '''
    A centered TextInput.
    '''

    text_width = NumericProperty()
    '''The text width
    '''

    def update_padding(self, *args):
        '''
        Update the padding so the text is centered
        '''
        if len(self.text) > 0:
            charFreeStr = ''.join(ch for ch in self.text if ch.isdigit() or ch == '.' or ch == ',')
            self.text = format(int(float(locale.atof(charFreeStr))), ",.0f")
        self.text_width = self._get_text_width(
            self.text,
            self.tab_width,
            self._label_cached
        )


class SetupOptions:
    def __init__(self):
        self.showPlots = False
        self.continueTraining = True
        self.modelType = "maskrcnn"
        self.numClasses = 1
        self.folderSuffix = "output"
        self.totalIterations = 10000
        self.iterationCheckpointPeriod = 1000
        self.validationDictPath = ''
        self.trainDictPath = ''
        self.rawImagesPath = ''


class SetupUI(MDApp):
    def __init__(self, **kwargs):
        self.title = "Deep Learning Training Setup UI"
        self.theme_cls.primary_palette = "Blue"
        super().__init__(**kwargs)
        self.root = Factory.SetupUI()
        modelTypeMenu_items = [{"text": "MaskRCNN"}, {"text": "PointRend"}]
        # technically grows always from the bottom left (closest to (0,0)), I think there is possibly something in menu.py or animation.py, especially related to
        # ver_growth/hor_growth, tar_x, tar_y, _start_coords, and anim=Animation difference of Auto and center/bottom cases and _animated_properties in animation.py
        # Or with MouseMotionEvent, its getting the 410 190 position and sending that into the animation, which I believe is the bottom left corner of the final widget
        # The commented lines in menu.py def open get closer, opens from the top left corner, but no items present in the menu
        self.modelTypeMenu = MDDropdownMenu(
            caller=self.root.ids['modelTypeButton'],
            items=modelTypeMenu_items,
            position="auto",
            callback=self.set_model,
            width_mult=4,
        )

        trueFalseMenu_items = [{"text":"True"}, {"text":"False"}]
        self.showPlotsMenu = MDDropdownMenu(
            caller=self.root.ids['showPlotsButton'],
            items=trueFalseMenu_items,
            position="auto",
            callback=self.set_showPlots,
            width_mult=4,
        )
        self.continueTrainingMenu = MDDropdownMenu(
            caller=self.root.ids['continueTrainingButton'],
            items=trueFalseMenu_items,
            position="auto",
            callback=self.set_continueTraining,
            width_mult=4,
        )

    def file_manager_open(self, destinationField, openType, message):
        print("use getFileOrDirList() here to launch file manager, then but text in a neighboring text field. Could also grab text from that field as an initialdir")
        listName = getFileOrDirList(openType, message, '.txt')
        self.root.ids[destinationField].text = listName.replace(os.path.expanduser('~'), '~')

    # def file_manager_open(self, destinationField):
    #     print("use getFileOrDirList() here to launch file manager, then but text in a neighboring text field. Could also grab text from that field as an initialdir")
    #     annotationTrainListFileName = getFileOrDirList('file', 'Annotation Train Dict List in text file', '.txt')
    #     self.root.ids['trainAnnotationDictPath'].text = annotationTrainListFileName

    def set_model(self, instance):
        self.root.ids['modelTypeButton'].set_item(instance.text)
        self.modelTypeMenu.dismiss()

    def set_showPlots(self, instance):
        self.root.ids['showPlotsButton'].set_item(instance.text)
        self.showPlotsMenu.dismiss()

    def set_continueTraining(self, instance):
        self.root.ids['continueTrainingButton'].set_item(instance.text)
        self.continueTrainingMenu.dismiss()

    def build(self):
        return self.root

    def on_start(self):
        # print("\non_start:")
        self.root.ids['fileManager_Train'].ids['lbl_txt'].halign = 'center'
        self.root.ids['fileManager_Validate'].ids['lbl_txt'].halign = 'center'
        self.root.ids['fileManager_Images'].ids['lbl_txt'].halign = 'center'


        store = JsonStore('SavedSetupOptions.json')
        if store.count() > 0:
            for key in store:
                if key in self.root.ids:
                    entry = self.root.ids[key]
                    print("\tid={0}, obj={1}".format(key, entry))
                    if isinstance(entry, MDTextField):
                        if key == 'rawImagesPath' or key == 'validateAnnotationDictPath' or key == 'trainAnnotationDictPath':
                            entry.text = store.get(key)['text'].replace(os.path.expanduser('~'), '~')
                        else:
                            entry.text = store.get(key)['text']
                        # print("\t\ttext=", entry.text)
                    elif isinstance(entry, MDDropDownItem):
                        entry.set_item(store.get(key)['text'])
                        # print("\t\tvalue=", entry.current_item)

    def on_stop(self):
        # print("\non_stop:")
        store = JsonStore('SavedSetupOptions.json')
        for key in self.root.ids:
            entry = self.root.ids[key]
            if isinstance(entry, MDTextField):
                # print("\tid={0}, text={1}".format(key, entry.text))
                if key == 'rawImagesPath' or key == 'validateAnnotationDictPath' or key == 'trainAnnotationDictPath':
                    store.put(key, text=entry.text.replace('~', os.path.expanduser('~')))
                else:
                    store.put(key, text=entry.text)
            elif isinstance(entry, MDDropDownItem):
                # print("\tid={0}, current_item={1}".format(key, entry.current_item))
                store.put(key, text=entry.current_item)

            if key == 'showPlotsButton':
                setupoptions.showPlots = textToBool(entry.current_item)
            elif key == 'continueTrainingButton':
                setupoptions.continueTraining = textToBool(entry.current_item)
            elif key == 'numClassesField':
                setupoptions.numClasses = int(entry.text)
            elif key == 'folderSuffixField':
                setupoptions.folderSuffix = entry.text
            elif key == 'modelTypeButton':
                setupoptions.modelType = entry.current_item
            elif key == 'iterationCheckpointPeriod':
                setupoptions.iterationCheckpointPeriod = entry.text
            elif key == 'totalIterations':
                setupoptions.totalIterations = entry.text
            elif key == 'trainAnnotationDictPath':
                setupoptions.trainDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'validateAnnotationDictPath':
                setupoptions.validationDictPath = entry.text.replace('~', os.path.expanduser('~'))
            elif key == 'rawImagesPath':
                setupoptions.rawImagesPath = entry.text.replace('~', os.path.expanduser('~'))

        self.root_window.close()


def setConfigurator(outputModelFolder: str = 'model', continueTraining: bool = False, baseStr: str = '', modelType: str = 'maskrcnn', numClasses: int = 1, maskType: str = 'polygon'):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(os.getcwd(), 'OutputModels', outputModelFolder)

    if modelType.lower() == 'pointrend':
        print('pointrend')
        # See params here https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend/point_rend/config.py
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(os.path.join(os.getcwd(), "detectron2_repo", "projects", "PointRend", "configs", "InstanceSegmentation", "pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = numClasses  # PointRend has to match num classes
        if continueTraining:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # If continuing training
        else:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final_3c3198.pkl')
    else:
        print('maskrcnn')
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        if continueTraining:
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
    cfg.SOLVER.MAX_ITER = 150000  # balloon test used 300 iterations, likely need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512, balloon test used 128)

    cfg.INPUT.MIN_SIZE_TRAIN = (1179,)  # (default: (800,))
    cfg.INPUT.MAX_SIZE_TRAIN = 1366  # (default: 1333)
    cfg.TEST.DETECTIONS_PER_IMAGE = 200  # Increased from COCO default, should never have more than 200 wires per image (default: 100)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000  # (default: 12000)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # (default: 6000)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

# maybe can check the dict and determine masktype from the segmentation entry without needing the separate field?
def fileHandling(annotationFileName):
    with open(annotationFileName, 'rb') as handle:
        fileContents = pickle.loads(handle.read())
        try:
            annotationDicts, maskType = fileContents
        except ValueError as e:
            if str(e) == 'too many values to unpack (expected 2)':
                # Using old version of dict
                maskType = 'polygon'
                annotationDicts = fileContents
            else:
                raise
    return annotationDicts, maskType


def setDatasetAndMetadata(baseStr: str, setupoptions: SetupOptions):
    showPlots = setupoptions.showPlots

    annotationTrainListFileName = setupoptions.trainDictPath
    annotationValidateListFileName = setupoptions.validationDictPath
    inputDirectoryName = setupoptions.rawImagesPath

    # Need to make a train and a validation list of dicts separately in InstanceSegmentationDatasetDict
    annotationTrainDicts, maskType = fileHandling(annotationTrainListFileName)
    annotationValidateDicts, altMaskType = fileHandling(annotationValidateListFileName)

    assert maskType == altMaskType, "The stated mask type from the Train and Validation annotation dicts do not match"
    assert maskType.lower() == 'bitmask' or maskType.lower() == 'polygon', "The valid maskType options are 'bitmask' and 'polygon'"

    annotationDicts = [annotationTrainDicts, annotationValidateDicts]

    # dirnames should return ['Train', 'Validation']
    (dirpath, dirnames, rawFileNames) = next(os.walk(inputDirectoryName))
    if 'Train' not in dirnames or 'Validation' not in dirnames:
        print('You are missing either a Train or Validation directory')
        quit()
    dirnames = ['Train',
                'Validation']  # After making sure these are directories as expected, lets force the order to match the annotationDicts order

    for d in range(len(dirnames)):
        if baseStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
            DatasetCatalog.register(baseStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
        MetadataCatalog.get(baseStr + "_" + dirnames[d]).set(thing_classes=["VerticalNanowires"])

    if showPlots:
        nanowire_metadata = MetadataCatalog.get(baseStr + "_Train")
        for d in random.sample(annotationTrainDicts, 20):
            fig, ax = plt.subplots(figsize=(10, 8))
            print(d["file_name"])
            rawImage = Image.open(d["file_name"])
            npImage = np.array(rawImage)
            visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
            visTest = visualizerNP.draw_dataset_dict(d)
            ax.imshow(visTest.get_image()[:, :, ::-1])
            plt.show(block=True)
    return maskType


def getFileOrDirList(fileOrFolder: str = 'file', titleStr: str = 'Choose a file', fileTypes: str = None):
    root = Tk()
    root.withdraw()
    assert fileOrFolder.lower() == 'file' or fileOrFolder.lower() == 'folder', "Only file or folder is an allowed string choice for fileOrFolder"
    if fileOrFolder.lower() == 'file':
        fileOrFolderList = filedialog.askopenfilename(initialdir=os.getcwd(), title=titleStr, filetypes=[(fileTypes + "file", fileTypes)])
    else:  # Must be folder from assert statement
        fileOrFolderList = filedialog.askdirectory(initialdir=os.getcwd(), title=titleStr)
    if not fileOrFolderList:
        quit()
    root.destroy()
    return fileOrFolderList


def textToBool(text):
    assert text.lower() == 'true' or text.lower() == 'false', "The passed text is not true/false"
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False


def main(setupoptions: SetupOptions):
    modelType = setupoptions.modelType # options are 'PointRend' or 'MaskRCNN'
    continueTraining = setupoptions.continueTraining
    outputModelFolder = modelType+"Model_" + setupoptions.folderSuffix
    numClasses = setupoptions.numClasses  # only has one class (VerticalNanowires)
    baseStr = 'VerticalNanowires'

    maskType = setDatasetAndMetadata(baseStr, setupoptions)
    configurator = setConfigurator(outputModelFolder, continueTraining, baseStr, modelType, numClasses, maskType)
    trainer = DefaultTrainer(configurator)
    trainer.resume_or_load(resume=continueTraining)
    trainer.train()


if __name__ == "__main__":
    setupoptions = SetupOptions()
    Window.size = (750, 725)
    Builder.load_file(f"TrainNewDataUI.kv")
    SetupUI().run()
    main(setupoptions)