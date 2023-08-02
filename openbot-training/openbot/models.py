# Created by Matthias Mueller - Intel Intelligent Systems Lab - 2020

from tkinter import W
import tensorflow as tf
import numpy as np
#from . import vit as vt

"""
Constructors for standard MLPs and CNNs
"""


def create_cnn(
    width,
    height,
    depth,
    cnn_filters=(8, 12, 16, 20, 24),
    kernel_sz=(5, 5, 5, 3, 3),
    stride=(2, 2, 2, 2, 2),
    padding="same",
    activation="relu",
    conv_dropout=0,
    mlp_filters=(64, 16),
    mlp_dropout=0.2,
    bn=False,
):

    # define input shape, channel dimension (tf convention: channels last) and img input
    inputShape = (height, width, depth)
    channelDim = -1
    inputs = tf.keras.Input(shape=inputShape, name="img_input")

    # build the cnn layer by layer
    for (i, f) in enumerate(cnn_filters):
        # set the input if it is the first layer
        if i == 0:
            x = inputs

        # build one block with conv, activation and optional bn and dropout
        x = tf.keras.layers.Conv2D(
            f,
            (kernel_sz[i], kernel_sz[i]),
            strides=(stride[i], stride[i]),
            padding=padding,
            activation=activation,
        )(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channelDim)(x)
        if conv_dropout > 0:
            x = tf.keras.layers.Dropout(conv_dropout)(x)

    # flatten output of the cnn and build the mlp
    x = tf.keras.layers.Flatten()(x)
    # build the mlp layer by layer
    for (i, f) in enumerate(mlp_filters):
        x = tf.keras.layers.Dense(f, activation=activation)(x)
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=channelDim)(x)
        if mlp_dropout > 0:
            x = tf.keras.layers.Dropout(mlp_dropout)(x)

    # assemble the model
    model = tf.keras.Model(inputs, x)

    # return the model
    return model


def create_mlp(in_dim, hidden_dim, out_dim, activation="relu", dropout=0.2):
    model = tf.keras.Sequential(name="MLP")
    model.add(
        tf.keras.layers.Dense(
            hidden_dim, input_dim=in_dim, activation=activation, name="cmd"
        )
    )
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(out_dim, activation=activation))
    return model


def pilot_net(img_width, img_height, bn=False):
    mlp = create_mlp(1, 1, 1, dropout=0)
    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(24, 36, 48, 64, 64),
        kernel_sz=(5, 5, 5, 3, 3),
        stride=(2, 2, 2, 1, 1),
        padding="valid",
        activation="relu",
        mlp_filters=(1164, 100),
        mlp_dropout=0,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(50, activation="relu")(combinedInput)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(10, activation="relu")(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="pilot_net", inputs=(cnn.input, mlp.input), outputs=x)
    return model


def cil_mobile(img_width, img_height, bn=True):
    mlp = create_mlp(1, 16, 16, dropout=0.5)
    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 64, 96, 128, 256),
        kernel_sz=(5, 3, 3, 3, 3),
        stride=(2, 2, 2, 2, 2),
        padding="same",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(128, 64),
        mlp_dropout=0.5,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(64, activation="relu")(combinedInput)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="cil_mobile", inputs=(cnn.input, mlp.input), outputs=x)

    return model


def cil_mobile_fast(img_width, img_height, bn=True):
    mlp = create_mlp(1, 16, 16)
    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 32, 64, 64, 128),
        kernel_sz=(5, 3, 3, 3, 2),
        stride=(2, 2, 2, 2, 2),
        padding="valid",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(512, 512),
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(64, activation="relu")(combinedInput)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.concatenate([mlp.input, x])
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(
        name="cil_mobile_fast", inputs=(cnn.input, mlp.input), outputs=x
    )

    return model


def cil(img_width, img_height, bn=True):
    mlp = create_mlp(1, 64, 64, dropout=0.5)
    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 32, 64, 64, 128, 128, 256, 256),
        kernel_sz=(5, 3, 3, 3, 3, 3, 3, 3),
        stride=(2, 1, 2, 1, 2, 1, 1, 1),
        padding="valid",
        activation="relu",
        conv_dropout=0.2,
        mlp_filters=(512, 512),
        mlp_dropout=0.5,
        bn=bn,
    )

    # fuse input MLP and CNN
    combinedInput = tf.keras.layers.concatenate([mlp.output, cnn.output])

    # output MLP
    x = tf.keras.layers.Dense(256, activation="relu")(combinedInput)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    # our final model will accept commands on the MLP input
    # and images on the CNN input, outputting two values (left/right ctrl)
    model = tf.keras.Model(name="cil", inputs=(cnn.input, mlp.input), outputs=x)

    return model


def simple_cnn(img_width, img_height, bn=False):
    cnn = create_cnn(
        img_width,
        img_height,
        3,
        cnn_filters=(32, 64, 128, 256),
        kernel_sz=(5, 3, 3, 3),
        stride=(2, 2, 2, 2),
        padding="same",
        activation="elu",
        conv_dropout=0.2,
        mlp_filters=(128, 64),
        mlp_dropout=0.5,
        bn=bn,
    )

    # output MLP
    #y = tf.keras.layers.MaxPool2D((9,9),(9,9),"valid")
    #cnn.input = y
    x = tf.keras.layers.Dense(16, activation="relu")(cnn.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear", dtype='float32')(x)

    # our final model will accept images on the CNN input, 
    cnn.summary()
    # outputting two values (throttle/steer ctrl)
    model = tf.keras.Model(name="behavior_clone", inputs=cnn.inputs, outputs=x)

    return model


def mobilnet_block(x, filters, strides, bn=False):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    
    return x


def mobil_net(img_width, img_height, bn=False):
    input = tf.keras.layers.Input((img_height, img_width, 3), name="img_input")
    x = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)

    x = mobilnet_block(x, filters = 64, strides = 1, bn=bn)
    x = mobilnet_block(x, filters = 128, strides = 2, bn=bn)
    x = mobilnet_block(x, filters = 128, strides = 1, bn=bn)
    x = mobilnet_block(x, filters = 256, strides = 2, bn=bn)
    x = mobilnet_block(x, filters = 256, strides = 1, bn=bn)
    x = mobilnet_block(x, filters = 512, strides = 2, bn=bn)

    for _ in range (5):
        x = mobilnet_block(x, filters = 512, strides = 1, bn=bn)
    x = mobilnet_block(x, filters = 1024, strides = 2, bn=bn)
    x = mobilnet_block(x, filters = 1024, strides = 1, bn=bn)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense (units = 64, activation = "elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units = 16, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=2, activation="linear")(x)

    model = tf.keras.Model(name="mobil_net", inputs=input, outputs=x)
    return model


def resnet_50(img_width, img_height, bn=False):
    input = tf.keras.layers.Input((img_height, img_width, 3), name="img_input")
    resnet = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=input,
        input_shape=(img_height, img_width, 3),
        pooling="avg"
    )
    resnet.trainable = False
    x = tf.keras.layers.Flatten()(resnet.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear")(x)

    model = tf.keras.Model(name="resnet_50", inputs=input, outputs=x)
    return model


def yolo():
    pass

def mobilenet_ver2(img_width, img_height, bn=False):
    #base_model = tf.keras.applications.MobileNetV2(input_shape=(img_width,img_height,3), include_top = False, alpha=0.5, weights=None)
    # output MLP
    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=True, alpha=0.2, weights=None, classes=2, classifier_activation=None)
    """
    x = tf.keras.layers.Dense(16, activation="relu")(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear", dtype='float32')(x)

    # our final model will accept images on the CNN input, 
    # outputting two values (throttle/steer ctrl)
    model = tf.keras.Model(name="mNet_v2", inputs=base_model.input, outputs=x)
    """

    return base_model

def mobilenet_v3(img_width : int, img_height : int, bn: bool=False) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224,224,3),
        alpha=1.0,
        minimalistic=True,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        dropout_rate=0.4,
        include_preprocessing=False,
    )
    for i, layer in enumerate(base_model.layers):
        if i < 108:
            base_model.layers[i].trainable = False #don't retrain the entire network it's too big
        if 'BatchNorm' in layer.name:
            base_model.layers[i].trainable = False #don't use any batch_normalization, it ruins test results

    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(name='mobilenet_v3', inputs=base_model.input, outputs=x)
    return model

def vgg_19(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
        pooling=None,
        classifier_activation=None,
    )
    for i, layer in enumerate(base_model.layers):
        if i < 18:
            base_model.layers[i].trainable = False
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation="elu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(name='vgg_19', inputs=base_model.input, outputs=x)
    return model

def resnet(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    x = tf.keras.Input(shape=(224,224,3))
    y = tf.keras.layers.AveragePooling2D((2,2),(2,2),"valid")(x)
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_shape=(112,112,3),
        weights=None,
        classifier_activation=None,
    )
    z = base_model(y)
    zf = tf.keras.layers.GlobalAveragePooling2D()(z)
    zf2 = tf.keras.layers.Dense(1000, activation="linear")(zf)
    out = tf.keras.layers.Dense(2, activation='linear')(zf2)
    model = tf.keras.Model(name='resnet', inputs=x, outputs=out)
    return model

def vgg16_bn(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    base_model = tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        classifier_activation=None
    )
    x = tf.keras.layers.Dense(2, activation='linear')(base_model.output)
    model = tf.keras.Model(name='vgg16_bn', inputs=base_model.input, outputs=x)
    return model

def googlenet(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    base_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        input_shape=(224,224,3),
        weights=None
    )
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(2, activation='linear')(x)
    model = tf.keras.Model(name='googlenet', inputs=base_model.input, outputs=x)
    return model

def alexnet(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=11, strides=4, padding='same', activation='relu', input_shape=(224,224,3)),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear'),
    ])
    return model

#def vit(img_width : int, img_height : int, bn : bool = False) -> tf.keras.Model:
    #return vt.create_VisionTransformer(2)
