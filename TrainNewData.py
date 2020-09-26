import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import multiprocessing
import random
import locale

from Utility.Utilities import *
from Utility.TrainNewDataTkinterUI import setupOptionsUI

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2_repo.projects.PointRend import point_rend
from detectron2.utils.logger import setup_logger
setup_logger()

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


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
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # (default: 512, balloon test used 128)

    # cfg.INPUT.MIN_SIZE_TRAIN = (1179,)  # (default: (800,)) previously 1179 for tilted
    # cfg.INPUT.MAX_SIZE_TRAIN = 1366  # (default: 1333) previously 1366 for tilted
    cfg.TEST.DETECTIONS_PER_IMAGE = 2000  # Increased from COCO default, should never have more than 2000 wires per image (default: 100)
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
            # rawImage = Image.open(d["file_name"]).convert('L')
            rawImage = Image.open(d["file_name"])
            npImage = np.array(rawImage)
            try:
                visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
            except IndexError:
                npImage = np.expand_dims(npImage, axis=2)
                visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
            visTest = visualizerNP.draw_dataset_dict(d)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(visTest.get_image()[:, :, ::-1])
            plt.show(block=True)
    return maskType


def main(setupoptions: SetupOptions):
    baseStr = 'VerticalNanowires_noSn_16'

    maskType = setDatasetAndMetadata(baseStr, setupoptions)
    configurator = setConfigurator(setupoptions, baseStr, maskType)
    trainer = DefaultTrainer(configurator)
    trainer.resume_or_load(resume=setupoptions.continueTraining)
    trainer.train()


if __name__ == "__main__":
    setupoptions = setupOptionsUI()
    main(setupoptions)