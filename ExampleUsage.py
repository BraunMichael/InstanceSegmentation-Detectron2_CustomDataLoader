import numpy as np
import wget
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()
fig, ax = plt.subplots(figsize=(10, 8))


inputFileName = 'input.jpg'
if not path.isfile(inputFileName):
    wget.download('http://images.cocodataset.org/val2017/000000439715.jpg', out=inputFileName)
rawImage = Image.open(inputFileName)
npImage = np.array(rawImage)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(npImage)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
outputs["instances"].pred_classes
outputs["instances"].pred_boxes

# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(npImage[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
ax.imshow(v.get_image()[:, :, ::-1])
plt.show(block=True)
# plt.pause(2)
# plt.close()