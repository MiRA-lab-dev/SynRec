from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import logging
import random
import skimage.io
import skimage.transform
from scipy.ndimage import gaussian_filter, map_coordinates

# from libtiff import TIFF

def load_image_gt(dataset, image_id, augment=False):
    # Load image and mask
    image = skimage.io.imread(os.path.join(dataset, 'data\\'+str(image_id)))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    # image = skimage.transform.resize(image,(960,960),mode='wrap')*255

    # image = image[:, :, 0]
    # cv2.imshow('img', image.astype('uint8'))
    # cv2.waitKey(0)
    # if image.ndim == 3:
    #     image = image[:,:,0:1]

    mask = skimage.io.imread(os.path.join(dataset, 'label\\'+str(image_id)))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    # mask = (mask > 0)
    # mask = skimage.transform.resize(mask,(960,960),mode='wrap')*255*255


    # mask = mask[:, :, 0]
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # if mask.ndim == 3:
    #     mask = mask[:,:,0:1]
    image = image.astype('float32')
    mask = mask.astype('float32')
    # Random horizontal flips.
    if augment:
        if random.randint(0, 3) == 0:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        else:
            if random.randint(0, 2) == 0:
                image = np.flipud(image)
                mask = np.flipud(mask)
            else:
                image = image
                mask = mask

        if random.randint(0, 4) == 0:
            image = np.rot90(image, 3)
            mask = np.rot90(mask, 3)
        else:
            if random.randint(0, 3) == 1:
                image = np.rot90(image, 1)
                mask = np.rot90(mask, 1)
            else:
                if random.randint(0, 2) == 1:
                    image = np.rot90(image, 2)
                    mask = np.rot90(mask, 2)
                else:
                    image = image
                    mask = mask
        # if random.randint(0, 2) == 0:
        #     imshape = (image.shape[0], image.shape[1])
        #
        #     # rng = np.random.RandomState(None)
        #     np.random.seed(sync_seed)
        #     # if np.random.random() < 0.5:
        #     #     return image
        #     result = np.ones((imshape[0], imshape[1]), dtype='uint8')
        #     result *= 255
        #
        #     # Make random fields
        #     dx = np.random.uniform(-1, 1, imshape) * 100
        #     dy = np.random.uniform(-1, 1, imshape) * 100
        #     # Smooth dx and dy
        #     sdx = gaussian_filter(dx, sigma=10, mode='reflect')
        #     sdy = gaussian_filter(dy, sigma=10, mode='reflect')
        #     # Make meshgrid
        #     x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        #     # Distort meshgrid indices
        #     distinds = (y + sdy), (x + sdx)
        #     for i in range(imshape[0]):
        #         for j in range(imshape[1]):
        #             # 行号
        #             newY = int(distinds[0][i][j])
        #             # 列号
        #             newX = int(distinds[1][i][j])
        #             if newY < 0:
        #                 newY = 0
        #             else:
        #                 if newY > imshape[0] - 1:
        #                     newY = imshape[0] - 1
        #             if newX < 0:
        #                 newX = 0
        #             else:
        #                 if newX > imshape[1] - 1:
        #                     newX = imshape[1] - 1
        #             result[newY, newX] = image[i, j, 0]

        if np.random.random() < 0.3:
            muti_noise = np.random.normal(1, 0.001, (image.shape[0], image.shape[1]))
            image *= muti_noise
        if np.random.random() < 0.3:
            add_noise = np.random.normal(0, 0.01, (image.shape[0], image.shape[1]))
            image += add_noise
        # if np.random.randint(2, 5) == 3:
        #     data = image[:, :, 0:1].copy()
        #     old_min = data.min()
        #     old_max = data.max()
        #     scale = np.random.normal(0.5, 0.1)
        #     center = np.random.normal(1.2, 0.2)
        #     data = scale * (data - old_min) + 0.5 * scale * center * (old_max - old_min) + old_min
        #     image = np.concatenate((data, data, data), axis=2)

    return image, mask


def data_generator(dataset, shuffle=True, augment=True,
                   batch_size=1):
    b = 0  # batch item index
    image_index = -1
    # image_ids = np.copy(dataset.image_ids)
    # image_ids = np.arange(0, length, step=1)
    image_ids = os.listdir(os.path.join(dataset,'data'))
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            # print('image_id:'+str(image_id))
            image, gt_masks = load_image_gt(dataset, image_id, augment=augment)

            # data preprocess
            mean = np.mean(image)
            std = np.std(image)
            image -= mean
            image /= std
            # image -= 128
            # image /= 33
            gt_masks/=255



            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size, image.shape[0],image.shape[1], 1) ,dtype=np.float32)
                # batch_gt_masks = np.zeros(
                #     (batch_size, image.shape[0], image.shape[1], 2))
                batch_gt_masks = np.zeros(
                     (batch_size, image.shape[0], image.shape[1], 1))

            # If more instances than fits in the array, sub-sample from them.
            # Add to batch
            # batch_images[b,:,:,0] = image-128
            # image /= 255
            batch_images[b, :, :, 0] = image
            # gt_masks/=255
            # cv2.imshow('img',batch_images[b, :, :, 0].astype('uint8'))
            # gt_masks=1-gt_masks
            batch_gt_masks[b, :, :, 0] = gt_masks
            # batch_gt_masks[b,:,:,0] = 1-gt_masks
            # cv2.imshow('mask_train',(batch_gt_masks[b,:,:,0]*255).astype('uint8'))
            # cv2.waitKey(0)
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images]
                outputs = [batch_gt_masks]
                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                image_id))
            error_count += 1
            if error_count > 5:
                raise


# if __name__ == "__main__":
    # aug = myAugmentation()
    # aug.Augmentation()
    # aug.splitMerge()
    # aug.splitTransform()
    # mydata = dataProcess(1248, 1248)
    # mydata.create_train_data()
    # mydata.create_test_data()
    # imgs_train,imgs_mask_train = mydata.load_train_data()
    # print imgs_train.shape,imgs_mask_train.shape
