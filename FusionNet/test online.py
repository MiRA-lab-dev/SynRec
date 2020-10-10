from NetworkArchitecture import FusionNet
import numpy as np
import cv2
from data import data_generator
from data import load_image_gt
from skimage import io
import matplotlib.pyplot as plt
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if __name__ == '__main__':
    input_shape = (512, 512, 1)
    # #####################  train #########################
    model = FusionNet(input_shape=input_shape)
    model.load_weights('model.10-0.054.hdf5')

    ####load data
    network_height = 512
    network_width = 512
    overlap = 50
    scale = 2

    ImagePath = 'image'
    MaskPath = 'mask'
    for imagefile in os.listdir(ImagePath):
        image = io.imread(os.path.join(ImagePath, imagefile))

        height_ori, width_ori = image.shape[0:2]
        image = cv2.resize(image, (int(width_ori/scale), int(height_ori/scale)))
        # image = cv2.equalizeHist(image[:,:,0])
        # image = np.stack([image,image,image],axis=2)

        ###crop
        height, width = image.shape[0:2]
        stitch_mask = np.zeros(shape=image.shape[0:2], dtype='uint8')
        step_h = network_height-overlap
        step_w = network_width-overlap
        rows = (height-network_height)//step_h+1
        cols = (width-network_width)//step_w+1
        for i in range(rows+1):
            if i == rows:
                start_i = height - network_height
            else:
                start_i = i*step_h
            for j in range(cols+1):
                if j == cols:
                    start_j = width - network_width
                else:
                    start_j = j*step_w
                # crop_img = image[i*step_h:i*step_h+network_height, j*step_w:j*step_w+network_width, :]
                crop_img = image[start_i:start_i + network_height, start_j:start_j + network_width]
                # plt.figure()
                # plt.imshow(crop_img,cmap='gray')
                # data pre-processing
                crop_img = crop_img.astype('float64')
                mean = np.mean(crop_img)
                std = np.std(crop_img)
                crop_img -= mean
                crop_img /= std
                # max=np.max(crop_img)
                # min=np.min(crop_img)
                # new_crop=(crop_img-min)/(max-min)*255
                # plt.figure()
                # plt.imshow((new_crop*255).astype('uint8'),cmap='gray')
                test = np.zeros(shape=(1, 512, 512, 1), dtype='float32')
                test[0, :, :, 0] = crop_img
                re = model.predict(test, batch_size=1, verbose=1)
                predict_label = ((re[0, :, :, 0]* 255).astype('uint8') )
                # io.imsave('./result3/' + str(i).rjust(3, '0') + '.png', predict_label)
                # plt.figure()
                # plt.imshow(predict_label,cmap='gray')

                ###stitch
                stitch_mask[start_i:start_i + network_height, start_j:start_j + network_width] = np.maximum(predict_label, stitch_mask[start_i:start_i + network_height, start_j:start_j + network_width])
        ###save mask
        stitch_mask = cv2.resize(stitch_mask, (width_ori, height_ori), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(MaskPath + '\\' + imagefile, (stitch_mask).astype('uint8'))