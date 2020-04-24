import os
import matplotlib.pyplot as plt
import numpy as np
from os import path
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()
basePath = '/home/mbraun/NewIS'
inputFileName = 'TestImages/2020_02_27_MB0236_Center_007.jpg'


# If need/want to clear the registered datasets
# DatasetCatalog.clear()

showPlots = True

InputDirectoryName = os.path.join(basePath,'VerticalNanowires')
if not InputDirectoryName:
    quit()
# dirnames should return ['Train', 'Validation']
(dirpath, dirnames, rawFileNames) = next(os.walk(InputDirectoryName))

nanowireStr = 'VerticalNanowires'
nanowire_metadata = MetadataCatalog.get(nanowireStr + "_" + dirnames[0])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512, balloon test used 128)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (VerticalNanowires)
cfg.TEST.DETECTIONS_PER_IMAGE = 200  # Increased from COCO default, should never have more than 200 wires per image (default: 100)
cfg.OUTPUT_DIR = os.path.join(basePath,'output')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.DATASETS.TEST = (nanowireStr + "_" + dirnames[1],)
predictor = DefaultPredictor(cfg)

if not path.isfile(inputFileName):
    quit()
rawImage = Image.open(inputFileName)
npImage = np.array(rawImage)

if npImage.ndim < 3:
        if npImage.ndim == 2:
            # Assuming black and white image, just copy to all 3 color channels
            npImage = np.repeat(npImage[:,:, np.newaxis], 3, axis=2)
        else:
            print('The imported rawImage is 1 dimensional for some reason, check it out.')
            quit()

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
outputs = predictor(npImage)
# outputs["instances"].pred_classes
# outputs["instances"].pred_boxes

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(npImage[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(v.get_image()[:, :, ::-1])
plt.show(block=True)
