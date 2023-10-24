import argparse

import imagesize

import os

import subprocess

parser = argparse.ArgumentParser(description='MegaDepth Undistortion')

parser.add_argument(
    '--colmap_path', type=str,
    default='/usr/bin',
    help='path to colmap executable'
)
parser.add_argument(
    '--base_path', type=str,
    default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu',
    help='path to mavic_npu'
)

args = parser.parse_args()

sfm_path = os.path.join(
    args.base_path, 'query_reference_SfM_model_500'
)
base_depth_path = os.path.join(
    sfm_path, 'dense'
)
output_path = os.path.join(
    args.base_path, 'query_reference_Undistorted_SfM_500'
)

os.mkdir(output_path)

image_path = os.path.join(
    base_depth_path, 'images'
)

# Find the maximum image size in scene.
max_image_size = 0
for image_name in os.listdir(image_path):
    max_image_size = max(
        max_image_size,
        max(imagesize.get(os.path.join(image_path, image_name)))
    )

# Undistort the images and update the reconstruction.
subprocess.call([
    os.path.join(args.colmap_path, 'colmap'), 'image_undistorter',
    '--image_path', os.path.join(args.base_path, 'query_reference_images_500'),
    '--input_path', os.path.join(sfm_path, 'sparse', '0'),
    '--output_path',  output_path,
    '--max_image_size', str(max_image_size)
])

# Transform the reconstruction to raw text format.
sparse_txt_path = os.path.join(output_path, 'sparse-txt')
os.mkdir(sparse_txt_path)
subprocess.call([
    os.path.join(args.colmap_path, 'colmap'), 'model_converter',
    '--input_path', os.path.join(output_path, 'sparse'),
    '--output_path', sparse_txt_path,
    '--output_type', 'TXT'
])