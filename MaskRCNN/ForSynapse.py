import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
import skimage
import scipy.misc
from skimage import measure,color

####突触配置
class SynapseConfig(Config):

    # Give the configuration a recognizable name
    NAME = "synapse"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3  # 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + mitochondria

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 960 #height 960
    IMAGE_MAX_DIM = 960 #width   old 1216

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = ( 8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 800

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class SynapseDataset(utils.Dataset):

    def load_infos(self,count,imagepath,maskpath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("ultrastructures", 1, "synapse")
        files = os.listdir(imagepath)
        for i in range(count):
            self.add_image("ultrastructures",image_id=i, path=imagepath+files[i],
                           maskpath=maskpath+files[i])


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ultrastructures":
            return info["ultrastructures"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = skimage.io.imread(info['maskpath'])
        mask = mask[:,:]
        label = measure.label(mask,connectivity=2)
        newmask = np.zeros((mask.shape[0],mask.shape[1],label.max()),dtype='int32')
        for i in range(1,label.max()+1):
            temp = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype='uint8')
            temp[label==i] = 1
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img',temp*255)
            # cv2.waitKey(0)
            newmask[:,:,i-1] = temp 
        # rgb = color.label2rgb(label)
        # cv2.imshow('label',np.reshape(rgb,(1250,1250,3)))
        # cv2.waitKey(0)

        # class_ids = 1
        class_ids = np.ones(label.max(), dtype='int32')
        return newmask, class_ids

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")



# synapse config
config = SynapseConfig()
config.display()

dataset_train = SynapseDataset()
dataset_train.load_infos(285, '.\\train\data\\', '.\\train\label\\')
dataset_train.prepare()

# Validation dataset
dataset_val = SynapseDataset()
dataset_val.load_infos(50, '.\\val\\images\\', '.\\val\masks\\')
dataset_val.prepare()

# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# train set
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE ,
            epochs=15,
            layers="all")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 5,
            epochs=30,
            layers="all")



# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

class InferenceConfig(SynapseConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



for image_id in range(50):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r["rois"], r['masks'], r["class_ids"],
                                dataset_val.class_names, figsize=(8, 8))






# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_val.image_ids, 37)
APs = []
# for image_id in image_ids:
for image_id in range(174):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    # visualize.plot_precision_recall(AP, precisions, recalls)
print("mAP: ", np.mean(APs))


