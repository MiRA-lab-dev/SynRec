
from NetworkArchitecture import FusionNet
from keras.optimizers import SGD, Nadam,Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np
from PIL import Image, ImageEnhance
import math
import keras.backend as K
# from keras_extend.preprocessing.image import ImageDataGenerator, standardize, random_transform
from scipy.ndimage import gaussian_filter, map_coordinates
# from keras.layers import initializers
# import keras.backend as K
import cv2
from data import data_generator
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.backend.common import epsilon


def pad(imgs):
    # 64
    return np.lib.pad(imgs, ((0, 0), (64, 64), (64, 64), (0, 0)), mode='reflect')


def resize(imgs):
    num = int(imgs.shape[0])
    array_new = np.zeros((num, 256, 256, 1), dtype=imgs.dtype)
    for i in range(imgs.shape[0]):
        I = Image.fromarray(imgs[i, :, :, 0]).resize((256, 256))
        array_new[i, :, :, 0] = np.array(I)

    return array_new


def lr_schedule(epoch):
    init_lr = 0.01
    if epoch == 0:
        return init_lr
    else:
        return init_lr * math.pow(0.7, epoch)


def AddNoise(x, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    # new_seed = np.random.randint(0, 2147483647)
    # np.random.seed(new_seed)
    if np.random.random() < 0.6:
        noise = np.random.normal(0, 0.1, (x.shape[0], x.shape[1], x.shape[2]))
        x = x + noise
    return x


def elastic_transform_image(image, alpha, sigma, sync_seed=None, **kwargs):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    # assert image1.shape == image2.shape
    # Take measurements
    imshape = (image.shape[0], image.shape[1])

    # rng = np.random.RandomState(None)
    np.random.seed(sync_seed)
    if np.random.random() < 0.2:
        return image

    # Make random fields
    dx = np.random.uniform(-1, 1, imshape) * alpha
    dy = np.random.uniform(-1, 1, imshape) * alpha
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
    sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
    # Distort meshgrid indices
    distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
    # Map cooordinates from image to distorted index set
    transformedimage1 = map_coordinates(image[:, :, 0], distinds, mode='reflect').reshape(imshape)
    # transformedimage2 = map_coordinates(image2, distinds, mode='reflect').reshape(imshape)
    # cv2.imwrite('elas.png',transformedimage1)
    image[:, :, 0] = transformedimage1

    return image


def elastic_transform_mask(image, alpha, sigma, sync_seed=None, **kwargs):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    # assert image1.shape == image2.shape
    # Take measurements
    imshape = (image.shape[0], image.shape[1])

    # rng = np.random.RandomState(None)
    np.random.seed(sync_seed)
    if np.random.random() < 0.2:
        return image

    # Make random fields
    dx = np.random.uniform(-1, 1, imshape) * alpha
    dy = np.random.uniform(-1, 1, imshape) * alpha
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
    sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
    # Distort meshgrid indices
    distinds = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
    # Map cooordinates from image to distorted index set
    transformedimage1 = map_coordinates(image[:, :, 0], distinds, mode='reflect').reshape(imshape)
    # transformedimage2 = map_coordinates(image2, distinds, mode='reflect').reshape(imshape)
    # cv2.imwrite('elas.png',transformedimage1)
    temp = np.zeros(imshape)
    temp[transformedimage1 > 120] = 255
    # differ = temp-image[:,:,0]
    # cv2.imwrite('differ.png',differ)
    image[:, :, 0] = temp
    return image


def rotate(x, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    # new_seed = np.random.randint(0, 2147483647)
    # np.random.seed(new_seed)
    if np.random.random() < 0.25:
        rotate_x = Image.fromarray(x[:, :, 0]).transpose(Image.ROTATE_90)
        x[:, :, 0] = np.array(rotate_x)
    elif np.random.random() < 0.5:
        rotate_x = Image.fromarray(x[:, :, 0]).transpose(Image.ROTATE_180)
        x[:, :, 0] = np.array(rotate_x)
    elif np.random.random() < 0.75:
        rotate_x = Image.fromarray(x[:, :, 0]).transpose(Image.ROTATE_270)
        x[:, :, 0] = np.array(rotate_x)

    return x


def normalization(x, **kwargs):
    x = x / 255
    return x


def elastic_distort(image, sync_seed):
    imshape = (image.shape[0], image.shape[1])

    # rng = np.random.RandomState(None)
    np.random.seed(sync_seed)
    # if np.random.random() < 0.5:
    #     return image
    result = np.ones((imshape[0], imshape[1]), dtype='uint8')
    result *= 255

    # Make random fields
    dx = np.random.uniform(-1, 1, imshape) * 100
    dy = np.random.uniform(-1, 1, imshape) * 100
    # Smooth dx and dy
    sdx = gaussian_filter(dx, sigma=10, mode='reflect')
    sdy = gaussian_filter(dy, sigma=10, mode='reflect')
    # Make meshgrid
    x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
    # Distort meshgrid indices
    distinds = (y + sdy), (x + sdx)
    for i in range(imshape[0]):
        for j in range(imshape[1]):
            # 行号
            newY = int(distinds[0][i][j])
            # 列号
            newX = int(distinds[1][i][j])
            if newY < 0:
                newY = 0
            else:
                if newY > imshape[0] - 1:
                    newY = imshape[0] - 1
            if newX < 0:
                newX = 0
            else:
                if newX > imshape[1] - 1:
                    newX = imshape[1] - 1
            result[newY, newX] = image[i, j, 0]
    # result = gaussian_filter(result,sigma = 4)
    cv2.imshow('img', result)


def randomBrightness(x, sync_seed=None, **kwargs):
    """
          对图像进行颜色抖动
          :param image: PIL的图像image
          :return: 有颜色色差的图像image
    """
    np.random.seed(sync_seed)
    new_seed = np.random.randint(0, 2147483647)
    np.random.seed(new_seed)
    if np.random.random() > 0.6:
        return x
    x_rgb = np.zeros((x.shape[0],x.shape[1],3),dtype='uint8')
    x_rgb[:,:,0] = x[:,:,0]
    x_rgb[:,:,1] = x[:,:,0]
    x_rgb[:,:,2] = x[:,:,0]
    image_ = Image.fromarray(x_rgb)
    # random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    # color_image = ImageEnhance.Color(image_).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(60, 140) / 100.  # 随机因子
    brightness_image = ImageEnhance.Brightness(image_).enhance(random_factor)  # 调整图像的亮度
    # random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    # sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    x[:, :, 0] = np.array(brightness_image)[:,:,0]
    # cv2.imshow('img',x)
    # cv2.waitKey(0)
    return x


def randomContrast(x, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    new_seed = np.random.randint(0, 2147483647)
    np.random.seed(new_seed)
    if np.random.random() > 0.6:
        return x
    image_ = Image.fromarray(x[:, :, 0], 'L')
    random_factor = np.random.randint(70, 130) / 100.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(image_).enhance(random_factor)  # 调整图像对比度 0.7-1.3
    x[:, :, 0] = np.array(contrast_image)
    return x


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_crossentropy_loss(y_true,y_pred):
    _epsilon = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred_clip = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y1 = -1 * y_true * K.log(y_pred_clip)
    y2 = -1 * (1-y_true) * K.log(1-y_pred_clip)
    return K.mean(y1+y2, axis=-1)

def weighted_binary_crossentropy_loss(y_true,y_pred):
    _epsilon = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred_clip = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred_logits = tf.log(y_pred_clip / (1 - y_pred_clip))
    element_wise_loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_true,logits=y_pred_logits,pos_weight=10)
    return K.mean(element_wise_loss, axis=-1)

def weighted_mse_loss(y_true,y_pred):
    return K.mean(200 * K.square(y_pred - y_true), axis=-1)


if __name__ == '__main__':
    smooth = 1
    input_shape = (512, 512, 1)
    # #####################  train #########################
    model = FusionNet(input_shape=input_shape)
    sgd = SGD(lr=0.001, momentum=0.9)
    nadam = Adam(lr=0.001)
    model.load_weights('model.10-0.193.hdf5')
    model_checkpoint = ModelCheckpoint('model.{epoch:02d}-{val_loss:.3f}.hdf5', monitor='val_loss', verbose=1)
    # model_lrschedule = LearningRateScheduler(lr_schedule)
    # model.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


    ####### data generator　########
    train_dataset = 'train'
    val_dataset = 'val'
    train_generator = data_generator(train_dataset, shuffle=True, augment=True, batch_size=1)
    val_generator = data_generator(val_dataset, shuffle=True,batch_size=1, augment=False)

    model.fit_generator(generator=train_generator, steps_per_epoch=1000, epochs=20, verbose=1,
                        callbacks=[model_checkpoint],
                        validation_data=next(val_generator), validation_steps=100)

    # model.fit_generator(train_generator, steps_per_epoch=2000, epochs=30,
    #                     validation_data=[validation_X[0:400, :, :, :], validation_Y[0:400, :, :, :]],
    #                     callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1), model_checkpoint])
    #
    # model.save_weights('FusionNet_ISBI.h5')
