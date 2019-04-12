import utils.config as Config
import utils.model as model
from utils.pascal_voc import VOC
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features= {
                                           'label': tf.FixedLenFeature([],tf.string),
                                           'img_raw': tf.FixedLenFeature([],tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img,[Config.image_size, Config.image_size, 3])
    img = (tf.cast(img, tf.float32)/255.0-0.5)*2
    label = tf.decode_raw(features['label'],tf.float32)
    label = tf.reshape(label,[Config.cell_size,Config.cell_size,25])
    return img,label





