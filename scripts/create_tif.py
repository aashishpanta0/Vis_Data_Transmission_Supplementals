# this script was explicitly used to create a tif file as an input for real-esrgan model
import os
import numpy as np
import matplotlib.pyplot as plt
import OpenVisus as ov
import tifffile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

crop_y_start, crop_y_end = 336, 440
crop_x_start, crop_x_end = 940, 1174

db = ov.LoadDataset("http://atlantis.sci.utah.edu/mod_visus?dataset=nex-gddp-cmip6&cached=arco")
data = db.read(quality=-4)

cropped_data = data[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

tifffile.imwrite('tif_test_low.tif', cropped_data)
