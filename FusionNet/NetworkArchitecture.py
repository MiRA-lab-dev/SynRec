from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.layers import Input, Activation, BatchNormalization
from keras import backend as K
from keras.layers.merge import add
from keras.layers import Dropout


def conv_bn(x, filters, kernel_size, strides, padding):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    return x


def residual(x, filters, kernel_size, strides, padding):
    y = conv_bn(x, filters, kernel_size, strides, padding)
    y = conv_bn(y, filters, kernel_size, strides, padding)
    y = conv_bn(y, filters, kernel_size, strides, padding)
    return add([x, y])


def conv_res_conv_block(x, filters, kernel_size, strides, padding):
    conv1 = conv_bn(x, filters, kernel_size, strides=strides, padding=padding)
    res = residual(conv1, filters, kernel_size, strides=strides, padding=padding)
    conv2 = conv_bn(res, filters, kernel_size, strides=strides, padding=padding)

    return conv2


def FusionNet(input_shape):
    inputs = Input(shape=input_shape)
    # down1
    block1 = conv_res_conv_block(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    down1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1)
    # down2
    block2 = conv_res_conv_block(down1, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
    down2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
    # down3
    block3 = conv_res_conv_block(down2, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')
    down3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block3)
    # down4
    block4 = conv_res_conv_block(down3, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')
    down4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block4)

    # bridge
    bridge = conv_res_conv_block(down4, filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')

    # upscaling4
    upscore4 = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(bridge)
    merge4 = add([upscore4, block4])
    up4 = conv_res_conv_block(merge4, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')

    # upscaling3
    upscore3 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(up4)
    merge3 = add([upscore3, block3])
    up3 = conv_res_conv_block(merge3, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')

    # upscaling2
    upscore2 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(up3)
    merge2 = add([upscore2, block2])
    up2 = conv_res_conv_block(merge2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')

    # upscaling1
    upscore1 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(up2)
    merge1 = add([upscore1, block1])
    up1 = conv_res_conv_block(merge1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    # up1 = Dropout(rate=0.5)(up1)

    # input=256*256
    # upscore0 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(up1)
    # upscore0 = Dropout(rate=0.5)(upscore0)

    output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(up1)

    model = Model(inputs=[inputs], outputs=[output])
    return model








