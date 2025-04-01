import argparse
import numpy as np
import tensorflow as tf
import OpenVisus as ov
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # comment this line to use gpu

# Register custom layers (needed for model loading and testing)
@tf.keras.utils.register_keras_serializable()
class SubpixelUpsampling(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(SubpixelUpsampling, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super(SubpixelUpsampling, self).get_config()
        config.update({"scale": self.scale})
        return config

@tf.keras.utils.register_keras_serializable()
class ResizeToTarget(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width, method="bilinear", **kwargs):
        super(ResizeToTarget, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
        self.method = method

    def call(self, inputs):
        return tf.image.resize(inputs, (self.target_height, self.target_width), method=self.method)

    def get_config(self):
        config = super(ResizeToTarget, self).get_config()
        config.update({
            "target_height": self.target_height,
            "target_width": self.target_width,
            "method": self.method,
        })
        return config

def load_data_with_nan_replacement(timestep, quality):
    db = ov.LoadDataset("http://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6")
    data = db.read(time=int(timestep), quality=int(quality))
    data = np.nan_to_num(data, nan=0.0)
    return data

def edge_loss(y_true, y_pred):
    sobel_true = tf.image.sobel_edges(y_true)
    sobel_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(sobel_true - sobel_pred))

def hybrid_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    max_val = tf.reduce_max(y_true)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
    ed_loss = edge_loss(y_true, y_pred)
    return mse_loss + ssim_loss + 0.1 * ed_loss

