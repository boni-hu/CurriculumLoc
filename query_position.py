import pandas as pd
import numpy as np
import cv2
from ast import literal_eval
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

'''
WGS 84 / UTM zone 17N
'''

gsd = 300/500  # m/pix
def image_point2gps_point(keypoint_position: np.array, image_position: np.array):
    point_position = np.array([image_position[0]+(keypoint_position[0]-250)*gsd, image_position[1]+(250 - keypoint_position[1])*gsd])
    alt = np.zeros(keypoint_position.shape[1])
    point_position= np.vstack([*point_position,alt])
    return point_position

query_csv = '/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/query.csv'
ref_csv = '/home/a409/users/huboni/Projects/dataset/GPR_competition/round2/Val/reference.csv'
ps_file = '/home/a409/users/chenlin/VPR_huboni/Val/GPR_Dataset/VPR_results/coordinate.txt'
query_file = pd.read_csv(query_csv,index_col="name")
reference_file = pd.read_csv(ref_csv,index_col="name")

ref_urls = []
query_kps = []
ref_kps = []
with open(ps_file) as f:
    lines = f.readlines()
    lines = [i.strip() for i in lines]
    query_name = lines[0].split("/")[-1]
    query_utm = query_file.loc[query_name, ["easting", "northing"]]
    for key, line in enumerate(lines):
        j = key % 4
        if j == 1:
            query_kps.append(literal_eval(line))
        elif j == 2:
            ref_urls.append("/".join(line.split("/")[-2:]))
        elif j == 3:
            ref_kps.append(literal_eval(line))
# multi ref
# TODO 这里加上ransac  看一下ransac原理，看所有都投影可以用吗
ref_image_position = reference_file.loc[ref_urls,["easting","northing"]]
# print(ref_image_position)
ref_image_position = np.array([ref_image_position["easting"],ref_image_position["northing"]]).transpose()
# print(ref_image_position)
ref_kpts_position = []
# print("ref_kps[0]:",ref_kps[0])
for i,image_position in enumerate(ref_image_position):
    ref_kpts_position.append(image_point2gps_point(np.array(ref_kps[i]).transpose(),image_position))
ref_kpts_position = np.hstack(ref_kpts_position).transpose()

query_kps = [np.array(i) for i in query_kps]
query_kps_position = np.vstack(query_kps)

M, mask = cv2.findHomography(query_kps_position, ref_kpts_position, cv2.RANSAC,10.0)
# print(M,mask)

k=np.array([[345,0, 247.328],[0,345,245],[0,0,1]])
# print(ref_kpts_position,'\n',query_kps_position,k)
sucess, R_vec, t, inliers = cv2.solvePnPRansac(ref_kpts_position,query_kps_position,k,np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
r_w2c, _ = cv2.Rodrigues(R_vec)
t_w2c = t
r_c2w = np.linalg.inv(r_w2c)
t_c2w = -r_c2w@t_w2c
pinpoint_utm = t_c2w[:,0][:2]
print("uav_pinpoint:", pinpoint_utm)
pp_loss = np.linalg.norm(query_utm - pinpoint_utm)
r1_utm = reference_file.loc[ref_urls[0],["easting","northing"]]
r1_loss = np.linalg.norm(query_utm - r1_utm)
print("localization loss:", pp_loss)
print("recall@1 loss:", r1_loss)