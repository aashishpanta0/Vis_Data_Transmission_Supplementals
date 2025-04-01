import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tqdm import tqdm
import OpenVisus as ov

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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


def residual_block(x, filters):
    res = x
    x = Conv2D(filters, (3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters, (3,3), padding="same")(x)
    x = Add()([x, res])
    return Activation("relu")(x)

def denoising_block(x, filters=64):
    y = Conv2D(filters, (3,3), dilation_rate=2, padding="same", activation="relu")(x)
    y = Conv2D(filters, (3,3), dilation_rate=2, padding="same", activation="relu")(y)
    return Add()([x, y])

def build_model(channels, target_height, target_width, num_blocks=4, upscale_factor=2):
    inputs = Input(shape=(None, None, channels))
    
    x = Conv2D(256, (3,3), padding="same", activation="relu")(inputs)
    x = Conv2D(256, (3,3), padding="same", activation="relu")(x)
    
    x = Conv2D(64 * (upscale_factor ** 2), (3,3), padding="same", activation="relu")(x)
    x = SubpixelUpsampling(scale=upscale_factor)(x)
    
    skip = x
    for _ in range(num_blocks):
        x = residual_block(x, 64)
    
    x = Conv2D(64, (3,3), padding="same")(x)
    x = Add()([x, skip])
    x = denoising_block(x, 64)
    
    x = Conv2D(channels, (3,3), padding="same")(x)
    
    x = ResizeToTarget(target_height=target_height, target_width=target_width)(x)
    
    outputs = x
    return Model(inputs=inputs, outputs=outputs)

def main(args):
    epochs = args.epochs
    timesteps_list = list(range(1950 * 365, 2014 * 365, 5))
    qualities = [-1, -2, -3, -4, -5, -6, -7, -8]
    
    sample_timestep = timesteps_list[0]
    full_res_data = load_data_with_nan_replacement(sample_timestep, 0)
    if full_res_data.ndim == 2:
        full_res_data = np.expand_dims(full_res_data, axis=-1)
    target_height, target_width = full_res_data.shape[:2]
    channels = full_res_data.shape[-1]

    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        model = build_model(channels, target_height, target_width)
        model.compile(optimizer="adam", loss=hybrid_loss)
    
    training_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for t in tqdm(timesteps_list, desc=f"Epoch {epoch+1}/{epochs}"):
            quality = np.random.choice(qualities)
            low_res_data = load_data_with_nan_replacement(t, quality)
            full_res_data = load_data_with_nan_replacement(t, 0)
            
            if low_res_data.ndim == 2:
                low_res_data = np.expand_dims(low_res_data, axis=-1)
            if full_res_data.ndim == 2:
                full_res_data = np.expand_dims(full_res_data, axis=-1)
            
            input_tensor = tf.expand_dims(low_res_data, axis=0)
            target_tensor = tf.expand_dims(full_res_data, axis=0)
            
            loss_value = model.train_on_batch(input_tensor, target_tensor)
            epoch_losses.append(loss_value)
        
        avg_epoch_loss = np.mean(epoch_losses)
        training_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.6f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a super-resolution model for climate data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    main(args)
