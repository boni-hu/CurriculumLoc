import argparse

import imagesize

import numpy as np

import os

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str,
    default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu',
    help='path to mavic_npu'
)

parser.add_argument(
    '--output_path', type=str,
    default='/home/a409/users/huboni/Projects/dataset/TerraTrack/process_output_query_ref_500',
    help='path to the output directory'
)


args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

base_path = args.base_path

# megadepth TODO
base_depth_path = os.path.join(
    base_path, 'query_reference_SfM_model_500/dense/stereo'
)

base_undistorted_sfm_path = os.path.join(
    base_path, 'query_reference_Undistorted_SfM_500'
)

undistorted_sparse_path = os.path.join(
    base_undistorted_sfm_path, 'sparse-txt'
)

depths_path = os.path.join(
    base_depth_path, 'depth_bin_h5'
)


images_path = os.path.join(
    base_undistorted_sfm_path, 'images'
)


# Process cameras.txt
with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header

camera_intrinsics = {}
for camera in raw:
    camera = camera.split(' ')
    camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

# Process points3D.txt
with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header

points3D = {}  # 3d点 字典：{点id:x,y,z}
for point3D in raw:
    point3D = point3D.split(' ')
    points3D[int(point3D[0])] = np.array([
        float(point3D[1]), float(point3D[2]), float(point3D[3])
    ])
    
# Process images.txt
with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
    raw = f.readlines()[4 :]  # skip the header

image_id_to_idx = {}
image_names = []
raw_pose = []
camera = []
points3D_id_to_2D = []
n_points3D = []
for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])):
    image = image.split(' ')
    points = points.split(' ')

    image_id_to_idx[int(image[0])] = idx

    image_name = image[-1].strip('\n')
    image_names.append(image_name)
    # print("image_name:", image_name)

    raw_pose.append([float(elem) for elem in image[1 : -2]])  # image pose[ QW, QX, QY, QZ, TX, TY, TZ]
    camera.append(int(image[-2]))
    current_points3D_id_to_2D = {}
    for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
        if int(point3D_id) == -1: # 3d点id为-1的跳过，说明图像中该2d点没有对应的3d点
            continue
        current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
    points3D_id_to_2D.append(current_points3D_id_to_2D) # 3d点的id与对应图像中2d点的坐标
    n_points3D.append(len(current_points3D_id_to_2D))
n_images = len(image_names)

# Image and depthmaps paths
image_paths = []
depth_paths = []
for image_name in image_names:
    image_path = os.path.join(images_path, image_name)
   
    # Path to the depth file
    depth_path = os.path.join(
        depths_path, image_name + '.geometric.h5'
    )
    # print("depth_path:", depth_path)

    if os.path.exists(depth_path):
        # Check if depth map or background / foreground mask
        depth_paths.append(depth_path)
        print("depth path:", depth_path)
        image_paths.append(image_path)
        print("image path:", image_path)
    else:
        depth_paths.append(None)
        image_paths.append(None)

# Camera configuration
intrinsics = []
poses = []
principal_axis = []
points3D_id_to_ndepth = []
for idx, image_name in enumerate(image_names):
    if image_paths[idx] is None:
        intrinsics.append(None)
        poses.append(None)
        principal_axis.append([0, 0, 0])
        points3D_id_to_ndepth.append({})
        continue
    image_intrinsics = camera_intrinsics[camera[idx]]
    K = np.zeros([3, 3])
    K[0, 0] = image_intrinsics[2]
    K[0, 2] = image_intrinsics[4]
    K[1, 1] = image_intrinsics[3]
    K[1, 2] = image_intrinsics[5]
    K[2, 2] = 1
    intrinsics.append(K)

    image_pose = raw_pose[idx]
    qvec = image_pose[: 4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    principal_axis.append(R[2, :])
    t = image_pose[4 : 7]
    # World-to-Camera pose
    current_pose = np.zeros([4, 4])
    current_pose[: 3, : 3] = R
    current_pose[: 3, 3] = t
    current_pose[3, 3] = 1
    # Camera-to-World pose
    # pose = np.zeros([4, 4])
    # pose[: 3, : 3] = np.transpose(R)
    # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
    # pose[3, 3] = 1
    poses.append(current_pose)
    
    current_points3D_id_to_ndepth = {}
    for point3D_id in points3D_id_to_2D[idx].keys():
        p3d = points3D[point3D_id]
        current_points3D_id_to_ndepth[point3D_id] = (np.dot(R[2, :], p3d) + t[2]) / (.5 * (K[0, 0] + K[1, 1])) 
    points3D_id_to_ndepth.append(current_points3D_id_to_ndepth)
principal_axis = np.array(principal_axis)
# 相机的朝向？
angles = np.rad2deg(np.arccos(
    np.clip(
        np.dot(principal_axis, np.transpose(principal_axis)),
        -1, 1
    )
))

# Compute overlap score
overlap_matrix = np.full([n_images, n_images], -1.)
scale_ratio_matrix = np.full([n_images, n_images], -1.)
for idx1 in range(n_images):
    if image_paths[idx1] is None or depth_paths[idx1] is None:
        continue
    for idx2 in range(idx1 + 1, n_images):
        if image_paths[idx2] is None or depth_paths[idx2] is None:
            continue
        # FIXME 计算match按位&的话，如果两个对应的id的顺序不一直呢
        # 共同观测到的特征点对应的三维点的 ID
        matches = (
            points3D_id_to_2D[idx1].keys() &
            points3D_id_to_2D[idx2].keys()
        )
        min_num_points3D = min(
            len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
        )  # FIXME 计算两个图像之间的重叠率 这个参数并没有用到
        overlap_matrix[idx1, idx2] = len(matches) / len(points3D_id_to_2D[idx1])  # min_num_points3D
        overlap_matrix[idx2, idx1] = len(matches) / len(points3D_id_to_2D[idx2])  # min_num_points3D
        if len(matches) == 0:
            continue
        points3D_id_to_ndepth1 = points3D_id_to_ndepth[idx1]
        points3D_id_to_ndepth2 = points3D_id_to_ndepth[idx2]
        nd1 = np.array([points3D_id_to_ndepth1[match] for match in matches])
        nd2 = np.array([points3D_id_to_ndepth2[match] for match in matches])
        # FIXME 这个比例缩放倍率最小值的计算可以用来判断这两个点在相机坐标系下的物理尺寸比例是否接近，从而用于后续的相对姿态估计。
        min_scale_ratio = np.min(np.maximum(nd1 / nd2, nd2 / nd1))
        scale_ratio_matrix[idx1, idx2] = min_scale_ratio
        scale_ratio_matrix[idx2, idx1] = min_scale_ratio
print(overlap_matrix)
np.savez(
    os.path.join(args.output_path, os.path.basename(base_path)+'.npz'),
    image_paths=image_paths,
    depth_paths=depth_paths,
    intrinsics=intrinsics,
    poses=poses,
    overlap_matrix=overlap_matrix,
    scale_ratio_matrix=scale_ratio_matrix,
    angles=angles,
    n_points3D=n_points3D,
    points3D_id_to_2D=points3D_id_to_2D,
    points3D_id_to_ndepth=points3D_id_to_ndepth
)
