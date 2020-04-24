import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import torch, torchvision
import detectron2
import zipfile
import wget
from os import path
import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
import random
from tkinter import Tk, filedialog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        rawImage = Image.open(filename)
        width, height = rawImage.size
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


InputDirectoryName = 'balloon'
if not path.isdir(InputDirectoryName):
    zipFileName = wget.download('https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip')
    with zipfile.ZipFile(zipFileName, "r") as zip_ref:
        zip_ref.extractall()

dataset_dicts = get_balloon_dicts("balloon/train")
for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")

fig, ax = plt.subplots(figsize=(10, 8))
for d in random.sample(dataset_dicts, 3):
    rawImage = Image.open(d["file_name"])
    npImage = np.array(rawImage)
    visualizerNP = Visualizer(npImage[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    visTest = visualizerNP.draw_dataset_dict(d)
    ax.imshow(visTest.get_image()[:, :, ::-1])
    plt.show(block=False)
    plt.pause(1)
    plt.cla()
plt.close()
print("Good to here")

#untested
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("balloon_val", )
predictor = DefaultPredictor(cfg)

dataset_dicts = get_balloon_dicts("balloon/val")

fig, ax = plt.subplots(figsize=(10, 8))
for d in random.sample(dataset_dicts, 3):
    rawImage = Image.open(d["file_name"])
    npImage = np.array(rawImage)
    outputs = predictor(npImage)
    v = Visualizer(npImage[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    ax.imshow(v.get_image()[:, :, ::-1])
    plt.show(block=False)
    plt.pause(1)
    plt.cla()
plt.close()

evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "balloon_val")

inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test