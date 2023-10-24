import numpy as np
from PIL import Image
import warnings
import os
import h5py
import argparse


'''
convert colmap depth map from .bin to .png or to .h5
'''
warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

# camnum = 12
# fB = 32504;
min_depth_percentile = 2
max_depth_percentile = 98

parser = argparse.ArgumentParser(description='convert depth bin to array and save h5')

parser.add_argument(
    '--depthmapsdir', type=str,
    help='path to the origin colmap output depth',
    default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/query_reference_SfM_model_500/dense/stereo/depth_maps'
)
parser.add_argument(
    '--output_h5_path', type=str,
    help='path to the save h5 depth',
    default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/query_reference_SfM_model_500/dense/stereo/depth_bin_h5'
)
args = parser.parse_args()


if not os.path.exists(args.output_h5_path):
    os.mkdir(args.output_h5_path)

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        # print("width:", width)
        # print("height:", height)
        # print("channels:", channels)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
        # print(array.shape)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def bin2depth(inputdepth, output_h5_path):
    # depth_map = '0.png.geometric.bin'
    # print(depthdir)
    # if min_depth_percentile > max_depth_percentile:
    #     raise ValueError("min_depth_percentile should be less than or equal "
    #                      "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(inputdepth):
        raise FileNotFoundError("file not found: {}".format(inputdepth))

    # np.set_printoptions(threshold=np.inf)

    depth_map = read_array(inputdepth)
    # depth_map[depth_map<=0] = 0
    min_depth, max_depth = np.percentile(depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    # depth_map[depth_map<=0] = 0
    depth_map = np.nan_to_num(depth_map)
    # print("input_bin_path:", inputdepth)
    depth_map_shape = depth_map.shape[0]*depth_map.shape[1]
    if depth_map_shape<100:
        print("depth_map_shape < 480*480!!!! and is:", depth_map_shape)
        print("depth name:", inputdepth)
    print(np.any(depth_map<0))  # 深度图存在负数  存在0， 取值在0~1之间

    # save depth as h5 FIXME
    h5_path = os.path.join(output_h5_path, '.'.join(os.path.basename(inputdepth).split('.')[:-1])+'.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('depth', data=depth_map)

    # bin 2 png
    # min_depth, max_depth = np.percentile(depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])
    # depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    # depth_map[depth_map < min_depth] = min_depth
    # depth_map[depth_map > max_depth] = max_depth

    # maxdisp = fB / min_depth;
    # mindisp = fB / max_depth;
    # depth_map = (fB/depth_map - mindisp) * 255 / (maxdisp - mindisp);
    # depth_map = np.nan_to_num(depth_map*255) # nan全都变为0
    # depth_map = depth_map.astype(int)

    # image = Image.fromarray(np.uint8(depth_map)).convert('L')
    # image = image.resize((500, 500), Image.ANTIALIAS) # 保证resize为500*500
    # print(image)
    # print(np.array(image))
    # ouputdepth = os.path.join(outputdir, '.'.join(os.path.basename(inputdepth).split('.')[:-1])+'.png')
    # print(ouputdepth)
    # image.save(ouputdepth)


for depthbin in os.listdir(args.depthmapsdir):
    inputdepth = os.path.join(args.depthmapsdir, depthbin)
    if os.path.exists(inputdepth):
        bin2depth(inputdepth, args.output_h5_path)