"""
Created by Marcel Santos - Intel Intelligent Systems Lab - 2021
This script implements several routines for data augmentation.
"""
import tensorflow as tf


def augment_img(img):
    """Color augmentation

    Args:
      img: input image

    Returns:
      img: augmented image
    """
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img


def augment_cmd(cmd):
    """
    Command augmentation

    Args:
      cmd: input command

    Returns:
      cmd: augmented command
    """
    if not (cmd > 0 or cmd < 0):
        coin = tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.dtypes.float32)
        if coin < 0.25:
            cmd = -1.0
        elif coin < 0.5:
            cmd = 1.0
    return cmd


# def flip_sample(img, cmd, label):
#     coin = tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.dtypes.float32)
#     label_n = label
#     if coin < 0.5:
#         img = tf.image.flip_left_right(img)
#         cmd = -cmd
#         label_n = [label[0], -label[1]]
#     return img, cmd, label_n

def flip_sample(img, label):
    coin = tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.dtypes.float32)
    label_n = label
    if abs(label[1]) <= 0.1:
        if coin < 0.5:
            img = tf.image.flip_left_right(img)
            label_n = [label[0], -label[1]]
    return img, label_n

def perspective_transform(alp, shift, img, rangeX=(184,1096), rangeY=(0,513), endShape=(90,160)):
    """Performs perspective transformation on img with arbitrary bounding box
    defined by rangeX, rangeY. alp is how much the image bounding box in the image is shifted
    from left to right, shift is how much the camera coord axis is rotated

    Parameters:
    -----------
    alp   : int
        Horizontal translation amount
    
    shift : int
        camera coord y translation amount
    
    img : PIL.Image
        Image to be transformed
    
    rangeX: [int, int]
        [Min, Max] x values for bounding box of region of interest
    
    rangeY: [int,int]
        [Min, Max] y values for bounding box of region of interest"""
    target_height = rangeY[1] - rangeY[0]
    target_width = rangeX[1] - rangeX[0]
    offset_height = rangeY[0] + shift 
    offset_width = rangeX[0] + alp
    img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    img = tf.image.resize(img, endShape)
    return img

def translation(img, label, rangeY = (0,252), rangeX = (96,544), endShape = (90,160), max = 96):
    throttle, steering = label[0], label[1]
    scale = 0.25
    difference = tf.constant(500, dtype=tf.int32)
    r = 100.0 
    while difference > tf.constant(max, dtype=tf.int32) or difference < tf.constant(-max, dtype=tf.int32):
        #deltaY = tf.random.uniform(shape=[], minval=0, maxval=50, dtype=tf.int32)
        deltaY = tf.constant(0)     
        r = tf.random.truncated_normal(shape=[], mean=steering, stddev=scale)
        if tf.abs(r) > 1.0:
            continue
        diff = 300 * (steering - r)
        difference = tf.cast(diff, dtype=tf.int32)
    img = perspective_transform(difference, deltaY, img, rangeX, rangeY, endShape)
    steering = r  # The pixel to angle conversion is approximate
    label = [throttle, steering]
    return img, label


