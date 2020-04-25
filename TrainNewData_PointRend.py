import os
import pickle
import matplotlib.pyplot as plt
import numpy as np  # (pip install numpy)
from PIL import Image, ImageOps
from tkinter import Tk, filedialog

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
import sys; sys.path.insert(1, "/home/mbraun/NewIS/detectron2_repo/projects/PointRend")
import point_rend


ContinueTraining = True
showPlots = False
root = Tk()
root.withdraw()
annotationTrainListFileName = filedialog.askopenfilename(initialdir="/home/mbraun/NewIS", filetypes=[('Annotation Train Dict List in text file', '.txt')])
if not annotationTrainListFileName:
    quit()

annotationValidateListFileName = filedialog.askopenfilename(initialdir="/home/mbraun/NewIS", filetypes=[('Annotation Validate Dict List in text file', '.txt')])
root.destroy()
if not annotationValidateListFileName:
    quit()

# Need to make a train and a validation list of dicts separately in InstanceSegmentationDatasetDict
with open(annotationTrainListFileName, 'rb') as handle:
    annotationTrainDicts = pickle.loads(handle.read())

with open(annotationTrainListFileName, 'rb') as handle:
    annotationValidateDicts = pickle.loads(handle.read())

annotationDicts = [annotationTrainDicts, annotationValidateDicts]
InputDirectoryName = filedialog.askdirectory(initialdir="/home/mbraun/NewIS", title = "Select folder with Training and Validation folders")
if not InputDirectoryName:
    quit()
# dirnames should return ['Train', 'Validation']
(dirpath, dirnames, rawFileNames) = next(os.walk(InputDirectoryName))
if 'Train' not in dirnames or 'Validation' not in dirnames:
    print('You are missing either a Train or Validation directory')
    quit()
dirnames = ['Train', 'Validation']  # After making sure these are directories as expected, lets force the order to match the annotationDicts order

nanowireStr = 'VerticalNanowires'
for d in range(len(dirnames)):
    if nanowireStr + "_" + dirnames[d] not in DatasetCatalog.__dict__['_REGISTERED']:
        DatasetCatalog.register(nanowireStr + "_" + dirnames[d], lambda d=d: annotationDicts[d])
    MetadataCatalog.get(nanowireStr + "_" + dirnames[d]).set(thing_classes=["VerticalNanowires"])
nanowire_metadata = MetadataCatalog.get(nanowireStr + "_Train")


if showPlots:
    for d in random.sample(annotationTrainDicts, 20):
        fig, ax = plt.subplots(figsize=(10, 8))
        print(d["file_name"])
        rawImage = Image.open(d["file_name"])
        npImage = np.array(rawImage)
        visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=nanowire_metadata, scale=0.5)
        visTest = visualizerNP.draw_dataset_dict(d)
        ax.imshow(visTest.get_image()[:, :, ::-1])
        plt.show(block=True)
        # plt.pause(10)
        # plt.cla()
    # plt.close()

cfg = get_cfg()
cfg.OUTPUT_DIR = '/home/mbraun/NewIS/PointRendModel_4mask'
# See params here https://github.com/facebookresearch/detectron2/blob/master/projects/PointRend/point_rend/config.py
point_rend.add_pointrend_config(cfg)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (VerticalNanowires)
cfg.MODEL.POINT_HEAD.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_CLASSES  # PointRend has to match num classes
cfg.merge_from_file("/home/mbraun/NewIS/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")

if ContinueTraining:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # If continuing training
else:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final_3c3198.pkl')

cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.DATASETS.TRAIN = (nanowireStr + "_Train",)
cfg.DATASETS.TEST = (nanowireStr + "_Validation",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 50000    # balloon test used 300 iterations, likely need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512, balloon test used 128)

cfg.INPUT.MIN_SIZE_TRAIN = (1179,)  # (default: (800,))
cfg.INPUT.MAX_SIZE_TRAIN = 1366  # (default: 1333)
cfg.TEST.DETECTIONS_PER_IMAGE = 200  # Increased from COCO default, should never have more than 200 wires per image (default: 100)
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000  # (default: 12000)
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # (default: 6000)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
if ContinueTraining:
    trainer.resume_or_load(resume=True)  # Only if starting from a model checkpoint
trainer.train()