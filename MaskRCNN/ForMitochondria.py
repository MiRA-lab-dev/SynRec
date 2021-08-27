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
from skimage import measure,color



####线粒体配置
class MitochondriaConfig(Config):

    # Give the configuration a recognizable name
    NAME = "mitochondria"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + mitochondria

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 1024 #height
    # IMAGE_MAX_DIM = 1024 #width
    IMAGE_MIN_DIM = 1024 #height
    IMAGE_MAX_DIM = 1024 #width

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_SCALES = (64,)

    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 512
    TRAIN_ROIS_PER_IMAGE = 256

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class MitochondriaDataset(utils.Dataset):

    def load_infos(self, imagepath, maskpath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "mitochondria")
        filenames = os.listdir(imagepath)
        i = 0
        for file in filenames:
            self.add_image("shapes", image_id=i, path=os.path.join(imagepath,file),
                           maskpath=os.path.join(maskpath, file))
            i += 1


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = skimage.io.imread(info['maskpath'])
        if mask.ndim==3:
            mask=mask[:,:,0]
        mask = mask[:,:]/255
        label = measure.label(mask,connectivity=2)
        newmask = np.zeros((mask.shape[0],mask.shape[1],label.max()),dtype='int32')

        for i in range(1,label.max()+1):
            temp = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype='uint8')
            temp[label==i] = 1
            # cv2.imshow('img',temp*255)
            # cv2.waitKey(0)
            newmask[:,:,i-1] = temp
        # rgb = color.label2rgb(label)
        # cv2.imshow('label',np.reshape(rgb,(768,1024,3)))
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

config = MitochondriaConfig()
config.display()

#train dataset 300
dataset_train = MitochondriaDataset()
dataset_train.load_infos('.\\train\images\\', '.\\train\masks\\')
dataset_train.prepare()


dataset_val = MitochondriaDataset()
dataset_val.load_infos('.\\val\\images\\', '.\\val\masks\\')
dataset_val.prepare()

# image_ids = np.random.choice(dataset_train.image_ids, 20)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#         modellib.load_image_gt(dataset_train, config,
#                                image_id, augment=True, use_mini_mask=False)
#     visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                                 dataset_val.class_names, figsize=(8, 8))
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

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
            learning_rate=config.LEARNING_RATE,
            epochs=15,
            layers="all")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30,
            layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

class InferenceConfig(MitochondriaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = '9'

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




for image_id in range(80):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)


    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_bbox)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)




    results = model.detect([original_image], verbose=1)
    r = results[0]
    masks = r['masks']

    visualize.display_instances(original_image, r["rois"], r['masks'], r["class_ids"],
                                dataset_val.class_names, figsize=(8, 8))



# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
recalls_list = []
precisions_list = []
for image_id in range(80):
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
    recall, precision = utils.compute_recall(r['rois'], gt_bbox, iou=0.5)
    recalls_list.append(recall)
    precisions_list.append(precision)

print("mAP: ", np.mean(APs))
print("recalls_list: ", np.mean(recalls))
print("precisions_list: ", np.mean(precisions))





