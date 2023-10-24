import numpy as np
import h5py
import os
from PIL import Image

# convert depth from png to h5
depth_png_path = '/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/reference_SfM_model_500/dense/stereo/depth_maps_png'
output_h5_path = '/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/reference_SfM_model_500/dense/stereo/depth_bin_h5'
if not os.path.exists(output_h5_path):
    os.mkdir(output_h5_path)

for png_name in os.listdir(depth_png_path):
    print("png_name:", png_name)
    depth_png = Image.open(os.path.join(depth_png_path, png_name))
    print(depth_png)
    png_array = np.array(depth_png)
    print(png_array)

    # Save depth map as HDF5 file
    h5_path = os.path.join(output_h5_path, '.'.join(png_name.split('.')[:3])+'.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('depth', data=png_array)

## Read h5 depth
# for h5_file in os.listdir(output_h5_path):
#     print("h5 name:", h5_file)
#     depth_h5 = h5py.File(os.path.join(output_h5_path, h5_file), 'r')
#     depth = depth_h5['depth']
#     print("h5_depth:",depth)
#     depth_array = np.array(depth)
#     print("depth_array:", depth_array)
#     depth_h5.close()