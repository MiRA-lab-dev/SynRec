import numpy as np
import cv2
import os
# from ForRAW import inference_config
from MitoConfig import InferenceConfig
import model as modellib

MODEL_DIR = 'D:\keras\liuj\YY_mitochondria\Mask-RCNN\logs'

###############
##generate the segmentation result for a 7K*8K image, don't produce a intermediate result
###############

if __name__ == '__main__':
    ####load model
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model_path = model.find_last()[1]
    model.load_weights(model_path, by_name=True)
    ####load data
    network_height = 2048
    network_width = 2048
    overlap = 100
    scale = 1.3
    ImagePath = 'D:\keras\liuj\YY_mitochondria\\raw data\\1\\raw'
    MaskPath = 'D:\keras\liuj\YY_mitochondria\mask\\1'
    for imagefile in os.listdir(ImagePath):
        image = cv2.imread(os.path.join(ImagePath, imagefile))

        height_ori, width_ori = image.shape[0:2]
        image = cv2.resize(image, (int(width_ori/scale), int(height_ori/scale)))
        # image = cv2.equalizeHist(image[:,:,0])
        # image = np.stack([image,image,image],axis=2)

        ###crop
        height, width = image.shape[0:2]
        stitch_mask = np.zeros(shape=image.shape[0:2], dtype='uint8')
        step_h = network_height-overlap
        step_w = network_width-overlap
        rows = height//step_h
        cols = width//step_w
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
                crop_img = image[start_i:start_i + network_height, start_j:start_j + network_width, :]
                results = model.detect([crop_img], verbose=1)
                r = results[0]
                masks = r['masks']
                mask = np.zeros(shape=(network_height, network_width), dtype='uint8')
                if masks.shape[0]==2048:
                    # cv2.imwrite('.\predict\\' + str(image_id).zfill(4) + '.png', mask * 255)
                    # continue
                    for t in range(masks.shape[2]):
                        # mask += masks[:, :, t].astype('uint8')
                        mask = np.logical_or(mask, masks[:, :, t])
                        # cv2.imwrite('.\predict\\' + str(image_id).zfill(4) + '.png', mask * 255)
                ###stitch
                stitch_mask[start_i:start_i + network_height, start_j:start_j + network_width] = np.logical_or(mask, stitch_mask[start_i:start_i + network_height, start_j:start_j + network_width])
        ###save mask
        stitch_mask = cv2.resize(stitch_mask, (width_ori, height_ori), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(MaskPath + '\\' + imagefile, (stitch_mask*255).astype('uint8'))







