import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform
import math
import os
from os.path import exists
from os.path import join
# torch.cuda.current_device()
# torch.cuda._initialized = True


#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 20
'''
/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference_images/offset_0_None/000001.png
/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference_images/offset_0_None/000002.png
/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference_images/offset_0_None/000000.png
/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference_images/offset_20_North/000002.png
/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference_images/offset_40_South/000002.png
'''

#Test1
imgfile1 = '/home/a409/users/huboni/Paper/locally_global_match_pnp/fig/fig-matching/q000000.png'
imgfile2 = '/home/a409/users/huboni/Paper/locally_global_match_pnp/fig/fig-matching/r000000.png'
# imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'

start = time.perf_counter()

# read left image
image1_o = cv2.imread(imgfile1)
image1 = cv2.resize(image1_o, (224,224))
image2_o = cv2.imread(imgfile2)
image2 = cv2.resize(image2_o, (224,224))
print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

kps_left, sco_left, des_left = cnn_feature_extract(image1_o,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2_o,  nfeatures = -1)

print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()

#Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# 这里使用的是KTreeIndex配置索引，指定待处理核密度树的数量（理想的数量在1-16）。
search_params = dict(checks=40)
# 用它来指定递归遍历的次数。值越高结果越准确，但是消耗的时间也越多。实际上，匹配效果很大程度上取决于输入。
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_left, des_right, k=2)
matches_reverse = flann.knnMatch(des_right, des_left, k=2)


goodMatch = []
locations_1_to_use = []
locations_2_to_use = []

# 匹配对筛选
disdif_avg = 0
# 统计平均距离差 第一匹配点和第二匹配点的距离差均值
for m, n in matches:
    disdif_avg += n.distance - m.distance
    disdif_avg = disdif_avg / len(matches)

for m, n in matches:
    #自适应阈值
    # if n.distance > m.distance + disdif_avg:
    if n.distance > m.distance + disdif_avg and matches_reverse[m.trainIdx][1].distance > matches_reverse[m.trainIdx][0].distance+disdif_avg and matches_reverse[m.trainIdx][0].trainIdx == m.queryIdx:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
print('match num is %d' % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)

# Perform geometric verification using RANSAC.
_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)

print('Found %d inliers' % sum(inliers))

inlier_idxs = np.nonzero(inliers)[0]
#最终匹配结果
matches = np.column_stack((inlier_idxs, inlier_idxs))
print('whole time is %6.3f' % (time.perf_counter() - start0))

# save inliners 像素坐标
coordinate_out_dir = '/home/a409/users/chenlin/VPR_huboni/Val/GPR_Dataset/VPR_results/'
coordinate_out_ref = []
coordinate_out_query = []

dist = 0
for i in range(matches.shape[0]):
    idx1 = matches[i, 0]
    idx2 = matches[i, 1]
    # 计算两幅匹配图像最终匹配点对的像素距离
    dist += math.sqrt(((locations_1_to_use[idx1,0]-locations_2_to_use[idx2,0])**2)+((locations_1_to_use[idx1,1]-locations_2_to_use[idx2,1])**2))
avg_dist = dist/matches.shape[0]
print("avg_dist:", avg_dist)
#     coordinate_out_query.append([locations_1_to_use[idx1, 0], locations_1_to_use[idx1, 1]])
#     coordinate_out_ref.append([locations_2_to_use[idx2, 0], locations_2_to_use[idx2, 1]])

# 输出最终两幅匹配图像匹配点对坐标
# if not exists(coordinate_out_dir):
#     os.mkdir(coordinate_out_dir)
# out_file = join(coordinate_out_dir, 'coordinate.txt')
# print('Writing recalls to', out_file)
# with open(out_file, 'a+') as res_out:
#     res_out.write(imgfile1 + '\n' + str(coordinate_out_query) + '\n')
#     res_out.write(imgfile2 + '\n' + str(coordinate_out_ref) + '\n')

# Visualize correspondences, and save to file.
#1 绘制匹配连线
plt.rcParams['savefig.dpi'] = 500 #图片像素
plt.rcParams['figure.dpi'] = 500 #分辨率
plt.rcParams['figure.figsize'] = (3.0, 2.0) # 设置figure_size尺寸
_, ax = plt.subplots()
plotmatch.plot_matches(
        ax,
        image1_o,
        image2_o,
        locations_1_to_use,
        locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        plot_matche_points = False,
        matchline = True,
        matchlinewidth = 0.1)
ax.axis('off')
ax.set_title('')
# plt.show()
plt.savefig('/home/a409/users/huboni/Paper/locally_global_match_pnp/fig/fig-matching/0.png')