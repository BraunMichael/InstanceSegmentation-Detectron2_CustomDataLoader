import os
import pickle
import matplotlib.pyplot as plt
import numpy as np  # (pip install numpy)
from PIL import Image, ImageOps
from tkinter import Tk, filedialog
import multiprocessing
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
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


modelType = 'MaskRCNN'  # options are 'PointRend' or 'MaskRCNN'
continueTraining = False
outputModelFolder = modelType+"Model_4masklowres"
numClasses = 1  # only has one class (VerticalNanowires)
showPlots = False


def setConfigurator(outputModelFolder: str = 'model', continueTraining: bool = False, baseStr: str = '', modelType: str = 'maskrcnn', numClasses: int = 1):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = os.path.join(os.getcwd(), outputModelFolder)

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
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATASETS.TRAIN = (baseStr + "_Train",)
    cfg.DATASETS.TEST = (baseStr + "_Validation",)
    cfg.DATALOADER.NUM_WORKERS = multiprocessing.cpu_count()
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000  # balloon test used 300 iterations, likely need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512, balloon test used 128)

    # cfg.INPUT.MIN_SIZE_TRAIN = (1179,)  # (default: (800,))
    # cfg.INPUT.MAX_SIZE_TRAIN = 1366  # (default: 1333)
    cfg.TEST.DETECTIONS_PER_IMAGE = 200  # Increased from COCO default, should never have more than 200 wires per image (default: 100)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000  # (default: 12000)
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # (default: 6000)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def setDatasetAndMetadata(baseStr: str):
    annotationTrainListFileName = getFileOrDirList('file', 'Annotation Train Dict List in text file', '.txt')
    annotationValidateListFileName = getFileOrDirList('file', 'Annotation Validation Dict List in text file', '.txt')
    InputDirectoryName = getFileOrDirList('folder', "Select folder with Training and Validation folders")

    # Need to make a train and a validation list of dicts separately in InstanceSegmentationDatasetDict
    with open(annotationTrainListFileName, 'rb') as handle:
        annotationTrainDicts = pickle.loads(handle.read())
    with open(annotationValidateListFileName, 'rb') as handle:
        annotationValidateDicts = pickle.loads(handle.read())
    annotationDicts = [annotationTrainDicts, annotationValidateDicts]

    # dirnames should return ['Train', 'Validation']
    (dirpath, dirnames, rawFileNames) = next(os.walk(InputDirectoryName))
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


def main():
    baseStr = 'VerticalNanowires'
    setDatasetAndMetadata(baseStr)
    configurator = setConfigurator(outputModelFolder, continueTraining, baseStr, modelType, numClasses)
    trainer = DefaultTrainer(configurator)
    if continueTraining:
        trainer.resume_or_load(resume=True)  # Only if starting from a model checkpoint
    trainer.train()


if __name__ == "__main__":
    main()