import argparse
import numpy as np
import tensorflow as tf
import OpenVisus as ov
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def main(args):
    model = tf.keras.models.load_model(args.modelpath, custom_objects={'hybrid_loss': hybrid_loss})
    print(f"Model loaded from {args.modelpath}.")

    timestep = args.timestep
    # define regions of interest
    regions = {
        "Region 1": (336, 440, 940, 1174),
        "Region 2": (60, 210, 430, 620),
        "Region 3": (370,450,1000,1100)
    }

    # Define quality levels
    quality_levels = [0, -2, -4, -6, -8]
    n_cols = len(quality_levels)
    n_regions = len(regions)
    n_rows = n_regions * 2 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(f"Multi-Region Super-Resolution Evaluation: Original vs. Predicted Outputs at different Quality Levels", fontsize=20)

    for r_idx, (region_name, (y_start, y_end, x_start, x_end)) in enumerate(regions.items()):
        print(f"\nProcessing {region_name} with region coordinates: y=[{y_start},{y_end}], x=[{x_start},{x_end}]")
        gt_crop = load_data(timestep, quality=0, y_range=[y_start, y_end], x_range=[x_start, x_end])
        original_crops = {}
        predicted_crops = {}

        for q in quality_levels:
            if q == 0:
                orig_crop = gt_crop
            else:
                orig_crop = load_data(timestep, quality=q, y_range=[y_start, y_end], x_range=[x_start, x_end])
            original_crops[q] = orig_crop
            if q == 0:
                predicted_crops[q] = None
            else:
                low_full = load_data(timestep, quality=q)
                if low_full.ndim == 2:
                    low_full = np.expand_dims(low_full, axis=-1)
                low_full_exp = np.expand_dims(low_full, axis=0)
                pred_full = model.predict(low_full_exp)[0, ..., 0]
                pred_crop = pred_full[y_start:y_end, x_start:x_end]
                predicted_crops[q] = pred_crop

        for c_idx, q in enumerate(quality_levels):
            ax_orig = axes[r_idx*2, c_idx] if n_rows > 1 else axes[c_idx]
            ax_orig.imshow(original_crops[q], cmap='viridis', origin='lower')
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if r_idx == 0:
                ax_orig.set_title(f"Q = {q}", fontsize=14)
            if c_idx == 0:
                ax_orig.set_ylabel(f"{region_name}\nOriginal", fontsize=14)

            ax_pred = axes[r_idx*2 + 1, c_idx] if n_rows > 1 else None
            if ax_pred is not None:
                if q == 0:
                    pass

                else:
                    ax_pred.imshow(predicted_crops[q], cmap='viridis', origin='lower')
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])
                if c_idx == 0:
                    ax_pred.set_ylabel(f"{region_name}\nPredicted", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = f"./images/{timestep}_3region_comparison.png"
    plt.savefig(output_filename, dpi=400)
    print(f"\nSaved comparison image to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a single image with 6 rows and 5 columns showing 3 regions (each with original and predicted crops) "
                    "across multiple quality levels (columns: Q=0, -2, -4, -6, -8). For quality 0, only the original full-res crop is shown."
    )
    parser.add_argument("--timestep", type=int, required=True, help="Timestep to process (e.g., 2010*365).")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to the saved model.")
    args = parser.parse_args()
    main(args)
