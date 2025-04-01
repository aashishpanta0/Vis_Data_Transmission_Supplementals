import os
import argparse
import numpy as np
import tensorflow as tf
import OpenVisus as ov
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from utils import *



def load_data(timestep, quality, y_range=None, x_range=None):
    """
    Loads data for a given timestep and quality.
    If y_range and/or x_range are provided, only that region is loaded.
    """
    print(f"Loading data for timestep={timestep}, quality={quality}")
    db = ov.LoadDataset("http://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6&cached=arco")
    if y_range is not None or x_range is not None:
        data = db.read(time=timestep, quality=quality, y=y_range, x=x_range)
    else:
        data = db.read(time=timestep, quality=quality)
    data = np.nan_to_num(data, nan=0)
    return data

def calculate_psnr(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    max_pixel = max(np.max(original), np.max(predicted))
    psnr = 10 * np.log10((max_pixel**2) / mse) if mse > 0 else float('inf')
    return psnr

def compute_ssim(img, gt):

    img = img.astype(np.float32)
    gt = gt.astype(np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
        gt = gt[..., np.newaxis]
    max_val = np.max(gt)
    ssim_val = tf.image.ssim(gt, img, max_val=max_val).numpy()
    return ssim_val


def main(args):
    model = tf.keras.models.load_model(args.modelpath, custom_objects={'hybrid_loss': hybrid_loss})
    print(f"Model loaded from {args.modelpath}.")
    
    timestep = args.timestep
    quality = args.quality 

    crop_y_start, crop_y_end = 370, 450
    crop_x_start, crop_x_end = 1000, 1100

    full_res = load_data(timestep, quality=0)
    low_res = load_data(timestep, quality)
    
    low_res_exp = low_res[np.newaxis, ..., np.newaxis]
    
    predicted_full = model.predict(low_res_exp)[0, ..., 0]
    
    target_size = (full_res.shape[0], full_res.shape[1])
    nearest_full = tf.image.resize(low_res_exp, size=target_size, method="nearest").numpy()[0, ..., 0]
    bilinear_full = tf.image.resize(low_res_exp, size=target_size, method="bilinear").numpy()[0, ..., 0]
    bicubic_full = tf.image.resize(low_res_exp, size=target_size, method="bicubic").numpy()[0, ..., 0]
    
    im2 = Image.open('./compare_models/Real_ESRGAN/Real-ESRGAN/sr_image/tif_test_low_out.tif')
    realesrgan_img = np.array(im2)
    

    predicted_crop = predicted_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    bicubic_crop   = bicubic_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    bilinear_crop  = bilinear_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    nearest_crop   = nearest_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    ratio_y = low_res.shape[0] / full_res.shape[0]
    ratio_x = low_res.shape[1] / full_res.shape[1]
    low_crop_y_start = int(crop_y_start * ratio_y)
    low_crop_y_end   = int(crop_y_end * ratio_y)
    low_crop_x_start = int(crop_x_start * ratio_x)
    low_crop_x_end   = int(crop_x_end * ratio_x)
    low_res_crop = low_res[low_crop_y_start:low_crop_y_end, low_crop_x_start:low_crop_x_end]
    

    fig = plt.figure(figsize=(22, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.25, wspace=0.2)
    
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(full_res, cmap='viridis', origin='lower')
    rect = Rectangle((crop_x_start, crop_y_start), 
                     crop_x_end - crop_x_start, crop_y_end - crop_y_start, 
                     edgecolor='red', facecolor='none', linewidth=2)
    ax_full.add_patch(rect)
    ax_full.set_title("Original Full Resolution (with crop region)", fontsize=22)
    ax_full.axis('off')
    
    ax_low = fig.add_subplot(gs[1, 0])
    ax_low.imshow(low_res_crop, cmap='viridis', origin='lower')
    ax_low.set_title("Low Resolution (1/16 of the original)", fontsize=22)
    ax_low.axis('off')
    
    ax_model = fig.add_subplot(gs[1, 1])
    ax_model.imshow(predicted_crop, cmap='viridis', origin='lower')
    ax_model.set_title("Our Model", fontsize=22)
    ax_model.axis('off')
    
    ax_realesrgan = fig.add_subplot(gs[2, 0])
    ax_realesrgan.imshow(realesrgan_img, cmap='viridis', origin='lower')
    ax_realesrgan.set_title("Real-ESRGAN", fontsize=22)
    ax_realesrgan.axis('off')
    
    ax_bicubic = fig.add_subplot(gs[2, 1])
    ax_bicubic.imshow(bicubic_crop, cmap='viridis', origin='lower')
    ax_bicubic.set_title("Bicubic", fontsize=22)
    ax_bicubic.axis('off')
    
    ax_bilinear = fig.add_subplot(gs[3, 0])
    ax_bilinear.imshow(bilinear_crop, cmap='viridis', origin='lower')
    ax_bilinear.set_title("Bilinear", fontsize=22)
    ax_bilinear.axis('off')
    
    ax_nearest = fig.add_subplot(gs[3, 1])
    ax_nearest.imshow(nearest_crop, cmap='viridis', origin='lower')
    ax_nearest.set_title("Nearest", fontsize=22)
    ax_nearest.axis('off')
    
    # fig.suptitle(f"Timestep: {timestep} | Crop: x=[{crop_x_start},{crop_x_end}], y=[{crop_y_start},{crop_y_end}]", fontsize=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_filename = f'./images/{timestep}_revised_comparison.png'
    plt.savefig(output_filename)
    print(f"\nSaved revised comparison image to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare superresolution methods with a custom layout: Full-res original, low-res input, our model, Real-ESRGAN, Bicubic, Bilinear, and Nearest."
    )
    parser.add_argument("--timestep", type=int, required=True, help="Timestep to process (e.g., 2010*365).")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--quality", type=int, default=-4, help="Quality level for low-res input (e.g., -4).")
    args = parser.parse_args()
    main(args)
