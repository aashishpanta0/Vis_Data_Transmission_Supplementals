import argparse
import numpy as np
import tensorflow as tf
import OpenVisus as ov
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # comment this line to use gpu
print(device_lib.list_local_devices())

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

def load_data(timestep, quality):
    print(f"Loading data for timestep={timestep}, quality={quality}")
    db = ov.LoadDataset("http://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6&cached=arco")
    data = db.read(time=timestep, quality=quality)
    data = np.nan_to_num(data, nan=0)
    return data

def calculate_psnr(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    max_pixel = max(np.max(original), np.max(predicted))
    psnr = 10 * np.log10((max_pixel**2) / mse) if mse > 0 else float('inf')
    return psnr

def main(args):
    model_path = args.modelpath
    model = tf.keras.models.load_model(model_path, custom_objects={'hybrid_loss': hybrid_loss})
    print(f"Model loaded from {model_path}.")
    
    timestep = args.timestep
    low_res_data = load_data(timestep, quality=args.quality)
    low_res_data = low_res_data[np.newaxis, ..., np.newaxis]  # Shape: (1, H, W, 1)
    
    print("Predicting full-resolution output...")
    predicted_full_res = model.predict(low_res_data)
    predicted_full_res = predicted_full_res[0, ..., 0]  
    
    # Load the original full-resolution data.
    original_full_res = load_data(timestep, quality=0)
    
    mse = np.mean((predicted_full_res - original_full_res) ** 2)
    psnr = calculate_psnr(original_full_res, predicted_full_res)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
    
    plt.figure(figsize=(18,10))
    plt.subplot(1,3,1)
    plt.imshow(low_res_data[0, :, :, 0], cmap='viridis', origin='lower')
    plt.title("Input Low-Resolution Data")
    plt.subplot(1,3,2)
    plt.imshow(original_full_res, cmap='viridis', origin='lower')
    plt.title("Original Full-Resolution Data")
    plt.subplot(1,3,3)
    plt.imshow(predicted_full_res, cmap='viridis', origin='lower')
    plt.title("Predicted Full-Resolution Data")
    plt.suptitle(f"Timestep: {timestep} - MSE: {mse:.4f} - PSNR: {psnr:.2f} dB", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'./images/{timestep}_qual{args.quality}_comparison.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run full-resolution prediction for nex-gddp-cmip6.")
    parser.add_argument("--timestep", type=int, required=True, help="timestep to process (efor example, product of  2010*365).  data range from 1950*365 to 2014*365.. for other timestepds, specify field from historical to sspxyz")
    parser.add_argument("--quality", type=int, required=True, help="input data quality level (-1 to -8)")
    parser.add_argument("--modelpath", type=str, required=True, help="path to the saved model.")
    args = parser.parse_args()
    main(args)
