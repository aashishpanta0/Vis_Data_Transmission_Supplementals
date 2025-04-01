import argparse
import numpy as np
import tensorflow as tf
import OpenVisus as ov
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


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

def compute_lpips(original, predicted):

    orig = tf.convert_to_tensor(original, dtype=tf.float32)
    pred = tf.convert_to_tensor(predicted, dtype=tf.float32)
    if len(orig.shape) == 2:
        orig = tf.expand_dims(orig, axis=-1)
    if len(pred.shape) == 2:
        pred = tf.expand_dims(pred, axis=-1)
    if orig.shape[-1] == 1:
        orig = tf.image.grayscale_to_rgb(orig)
    if pred.shape[-1] == 1:
        pred = tf.image.grayscale_to_rgb(pred)
    orig = tf.expand_dims(orig, axis=0)
    pred = tf.expand_dims(pred, axis=0)
    
    orig = tf.keras.applications.vgg16.preprocess_input(orig)
    pred = tf.keras.applications.vgg16.preprocess_input(pred)
    
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    outputs = [vgg.get_layer(name).output for name in layer_names]
    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    
    orig_features = feature_extractor(orig)
    pred_features = feature_extractor(pred)
    
    lpips_score = 0.0
    for of, pf in zip(orig_features, pred_features):
        def normalize(feat):
            norm = tf.sqrt(tf.reduce_sum(tf.square(feat), axis=-1, keepdims=True) + 1e-10)
            return feat / norm
        of_norm = normalize(of)
        pf_norm = normalize(pf)
        diff = of_norm - pf_norm
        layer_score = tf.reduce_mean(tf.square(diff))
        lpips_score += layer_score
    return lpips_score.numpy()


def main(args):
    model = tf.keras.models.load_model(args.modelpath, custom_objects={'hybrid_loss': hybrid_loss})
    print(f"Model loaded from {args.modelpath}.")
    timestep = args.timestep

    crop_y_start, crop_y_end = 336, 440
    crop_x_start, crop_x_end = 940, 1174

    full_res = load_data(timestep, quality=0)
    
    quality_levels = [-2, -4, -6, -8]
    n_levels = len(quality_levels)
    total_cols = n_levels + 1

    fig = plt.figure(figsize=(5 * total_cols, 8))
    gs = gridspec.GridSpec(2, total_cols, width_ratios=[1] + [1]*n_levels)

    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(full_res, cmap='viridis', origin='lower')
    ax_full.set_title("Full Resolution (Quality 0)")
    ax_full.axis('off')
    rect = Rectangle((crop_x_start, crop_y_start), 
                     crop_x_end - crop_x_start, 
                     crop_y_end - crop_y_start, 
                     edgecolor='red', facecolor='none', linewidth=2)
    ax_full.add_patch(rect)
    
    for idx, quality in enumerate(quality_levels):
        print(f"\nProcessing quality level {quality}...")
        low_entire = load_data(timestep, quality)
        low_entire_exp = low_entire[np.newaxis, ..., np.newaxis]  
        
        predicted_full = model.predict(low_entire_exp)[0, ..., 0]
        
        factor_y = predicted_full.shape[0] / low_entire.shape[0]
        factor_x = predicted_full.shape[1] / low_entire.shape[1]
        print(f"Upscaling factors: factor_y = {factor_y:.2f}, factor_x = {factor_x:.2f}")

        # crop the predicted full-res image.
        predicted_crop = predicted_full[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        # crop the corresponding region from the low-res input.
        low_y_start = int(crop_y_start / factor_y)
        low_y_end   = int(crop_y_end / factor_y)
        low_x_start = int(crop_x_start / factor_x)
        low_x_end   = int(crop_x_end / factor_x)
        low_crop = low_entire[low_y_start:low_y_end, low_x_start:low_x_end]

        gt_crop = load_data(timestep, quality=0, y_range=[crop_y_start, crop_y_end], x_range=[crop_x_start, crop_x_end])
        mse = np.mean((predicted_crop - gt_crop) ** 2)
        psnr = calculate_psnr(gt_crop, predicted_crop)
        lpips = compute_lpips(gt_crop, predicted_crop)
        print(f"Quality {quality} -> Crop MSE: {mse:.4f}, PSNR: {psnr:.2f} dB, LPIPS: {lpips:.4f}")

        ax_low = fig.add_subplot(gs[0, idx + 1])
        ax_low.imshow(low_crop, cmap='viridis', origin='lower')
        ax_low.set_title(f"Low Res (Q={quality})")
        ax_low.axis('off')
        
        ax_pred = fig.add_subplot(gs[1, idx + 1])
        ax_pred.imshow(predicted_crop, cmap='viridis', origin='lower')
        ax_pred.set_title(f"Predicted Full Res\nMSE: {mse:.4f} | PSNR: {psnr:.2f} dB\nLPIPS: {lpips:.4f}")
        ax_pred.axis('off')
    
    # fig.suptitle(f"Timestep: {timestep} | Region: x=[{crop_x_start},{crop_x_end}], y=[{crop_y_start},{crop_y_end}]", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_filename = f'./images/region_comparison_with_fullres.png'
    plt.savefig(output_filename)
    print(f"\nSaved comparison image to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For qualities -2, -4, -6, -8: predict on the entire region then crop a fixed region. "
                    "The left panel shows full resolution (quality 0) with the selected region boxed."
    )
    parser.add_argument("--timestep", type=int, required=True, help="Timestep to process (e.g., 2010*365).")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to the saved model.")
    args = parser.parse_args()
    main(args)
