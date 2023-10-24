'''
this code can generate uav and matched sat dataset from uav and xian.geotiff
'''

from osgeo import gdal, osr
import cv2
import os
import exifread
import pandas as pd
import numpy as np
import argparse
import math

def getLatOrLng(refKey, tudeKey, tags):
    '''
    获取UAV经度或纬度
    '''
    if refKey not in tags:
        return None
    # ref = tags[refKey].printable
    LatOrLng = tags[tudeKey].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
    LatOrLng = float(LatOrLng[0]) + float(LatOrLng[1]) / 60 + float(LatOrLng[2]) / float(LatOrLng[3]) / 3600
    if refKey == 'GPS GPSLatitudeRef' and tags[refKey].printable != "N":
        LatOrLng = LatOrLng * (-1)
    if refKey == 'GPS GPSLongitudeRef' and tags[refKey].printable != "E":
        LatOrLng = LatOrLng * (-1)
    return LatOrLng

def save_query_img(uav_img_dir, query_img_dir, file):
    '''
    UAV images 裁剪缩放4000*3000 ->缩放 666*500 -> 裁剪500*500 保证变换前后坐标不变
    '''
    resize_size = 512
    org_uav = cv2.imread(os.path.join(uav_img_dir, file))
    height, width = org_uav.shape[0], org_uav.shape[1]
    if width >= height:
        left_p = int(((width/height)*resize_size-resize_size)/2)
        right_p = left_p + resize_size
        query = cv2.resize(org_uav, (int(width/height*resize_size), resize_size))[:, left_p:right_p]
        cv2.imwrite(os.path.join(query_img_dir, file), query)
    else:
        up_p = ((height/width)*resize_size-resize_size)/2
        down_p = ((height/width)*resize_size-resize_size)/2 + resize_size
        query = cv2.resize(org_uav, (resize_size,(height/width)*resize_size))[up_p:down_p, :]
        cv2.imwrite(os.path.join(query_img_dir, file), query)

    print("query images save done!")

def GetTifInfo(filename):
    print(" Reading Tiff...")
    dataset = gdal.Open(filename)
    tiff_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    # (12070367.603561446, 1.1943285669558756, 0.0, 4074404.7301915865, 0.0, -1.194328566955883)
    # osr.SpatialReference 提供描绘和转换坐标系统的服务 地理坐标(用经纬度表示)；投影坐标(如 UTM ，用米等单位量度来定位)。
    pcs = osr.SpatialReference()
    pcs.ImportFromWkt(dataset.GetProjection())
    gcs = pcs.CloneGeogCS()  # 地理空间坐标系
    # shape = (dataset.RasterYSize, dataset.RasterXSize) # (80769,77441)
    img_r = dataset.GetRasterBand(1)  # (height, width)
    img_g = dataset.GetRasterBand(2)
    img_b = dataset.GetRasterBand(3)
    print(" Read Tiff Done! ")
    return img_r, img_g, img_b, dataset, gcs, pcs, tiff_geotrans

# 经纬度转UTM坐标
def Lonlat2Xy(SourceGcs, TargetPcs, lon, lat):
    '''
    :param SourceRef: 源地理坐标系统
    :param TargetRef: 目标投影
    :param lon: 待转换点的longitude值
    :param lat: 待转换点的latitude值
    :return:
    '''
    # 创建目标空间参考
    spatialref_target=osr.SpatialReference()
    spatialref_target.ImportFromEPSG(TargetPcs)
    # 创建原始空间参考
    spatialref_source=osr.SpatialReference()
    spatialref_source.ImportFromEPSG(SourceGcs)  #4326 为原始空间参考的ESPG编号，WGS84
    # 构建坐标转换对象，用以转换不同空间参考下的坐标
    trans=osr.CoordinateTransformation(spatialref_source,spatialref_target)
    # coordinate_after_trans 是一个Tuple类型的变量包含3个元素， [0]为y方向值，[1]为x方向值，[2]为高度
    coordinate_after_trans=trans.TransformPoint(lat,lon)
    return coordinate_after_trans

def Mec2Lonlat(x, y):
    x = float(x)
    y = float(y)
    x = x / 20037508.34 * 180
    y = y / 20037508.34 * 180
    y = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
    return [x, y]

# 经纬度转tiff行列坐标
def Lonlat2Rowcol(tiff_geotrans, utm_x, utm_y):

    A = [[tiff_geotrans[1], tiff_geotrans[2]],
         [tiff_geotrans[4], tiff_geotrans[5]]]
    s = [[utm_x - tiff_geotrans[0]],  # 运用矩阵解二元一次方程组求得行列号
         [utm_y - tiff_geotrans[3]]]
    r = np.linalg.solve(A, s)
    Xpixel,Ypixel = int(r[0]), int(r[1])
    return Xpixel,Ypixel


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='TerraTrack dataset generation')
    parser.add_argument('--tiff_data', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/西安市.tif',
                        help='sat image origin domain google geotiff')
    parser.add_argument('--uav_img_dir', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/origin_uavs',
                        help='origin DroneMap uav images path')
    parser.add_argument('--save_path', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/',
                        help='Dir to save recall')
    parser.add_argument('--output_beacon', type=str, default='query',
                        help='output data type: query or reference')
    parser.add_argument('--offset_beacon', type=str, default='None',
                        help='None, West_20, West_40, East_20, East_40, North_20, North_40, South_20, South_40')

    opt = parser.parse_args()
    print(opt)

    if opt.output_beacon == 'query':
        '''
        save query_images and query.csv
        '''
        query_img_dir = os.path.join(opt.save_path, 'query_images')
        if not os.path.exists(query_img_dir):
            os.mkdir(query_img_dir)

        files = os.listdir(opt.uav_img_dir)
        name_l = []
        eas_l = []
        nor_l = []
        alt_l = []
        for file in files:
            print("uav image name:", file)
            img = open(os.path.join(opt.uav_img_dir, file), 'rb')
            tags = exifread.process_file(img)  # tags保存所有图像信息，已键-值保存
            uav_lon = getLatOrLng('GPS GPSLongitudeRef', 'GPS GPSLongitude', tags)  # 经度
            uav_lat = getLatOrLng('GPS GPSLatitudeRef', 'GPS GPSLatitude', tags)  # 纬度
            print('uav_lon：{0} uav_lat：{1}'.format(uav_lon, uav_lat))
            # 经纬度转UTM：4326 为原始空间参考的ESPG编号 32648为目标空间（西安UTM）参考的ESPG编号 3857为geotiff投影空间编号
            # uav_utm_xy = Lonlat2Xy(4326, 32648, uav_lon, uav_lat)
            uav_utm_xy = Lonlat2Xy(4326, 32648, uav_lon, uav_lat)
            print("uav utm_xy:", uav_utm_xy)

            # resize and save query images from orgin images
            save_query_img(opt.uav_img_dir, query_img_dir, file)

            # 生成query list 保存query.csv
            name_l.append(file)
            eas_l.append(uav_utm_xy[1])
            nor_l.append(uav_utm_xy[0])
            alt_l.append(float(tags['GPS GPSAltitude'].values[0]))

        # 保存query.csv
        query_dict = {"easting": eas_l,
                      "northing": nor_l,
                      "altitude": alt_l,
                      "name": name_l}
        pd.DataFrame(query_dict).to_csv(os.path.join(opt.save_path, "query.csv"), index=False)

    else:
        '''
        save reference_images and reference.csv, give some offest
        '''
        ref_img_dir = os.path.join(opt.save_path, 'reference_images_512')
        if not os.path.exists(ref_img_dir):
            os.mkdir(ref_img_dir)

        tiff_r, tiff_g, tiff_b, dataset, gcs, pcs, tiff_geotrans = GetTifInfo(opt.tiff_data)
        # 读取UAV images 并按照UAV的经纬度坐标裁剪保存对应的与偏移的遥感图像
        files = os.listdir(opt.uav_img_dir)
        name_l = []
        eas_l = []
        nor_l = []
        for file in files:
            print("uav image name:", file)
            img = open(os.path.join(opt.uav_img_dir, file), 'rb')
            tags = exifread.process_file(img)  # tags保存所有图像信息，已键-值保存
            uav_lon = getLatOrLng('GPS GPSLongitudeRef', 'GPS GPSLongitude', tags)  # 经度
            uav_lat = getLatOrLng('GPS GPSLatitudeRef', 'GPS GPSLatitude', tags)  # 纬度
            print('uav_lon：{0} uav_lat：{1}'.format(uav_lon, uav_lat))
            # 经纬度转UTM：4326 为原始空间参考的ESPG编号 32648为目标空间（西安UTM）参考的ESPG编号 3857为geotiff投影空间编号
            uav_mec_xy = Lonlat2Xy(4326, 3857, uav_lon, uav_lat)
            # 根据中心坐标计算左上角坐标 长宽分别为391.16m  X方向偏移 Y方向偏移 500=GSD*uav_width_pixel 这里更改为自适应 TODO
            if opt.offset_beacon == 'None':
                x_offset, y_offset = 0.0, 0.0
            elif opt.offset_beacon == 'West_20':
                x_offset, y_offset = -20.0, 0.0
            elif opt.offset_beacon == 'West_40':
                x_offset, y_offset = -40.0, 0.0
            elif opt.offset_beacon == 'East_20':
                x_offset, y_offset = 20.0, 0.0
            elif opt.offset_beacon == 'East_40':
                x_offset, y_offset = 40.0, 0.0
            elif opt.offset_beacon == 'North_20':
                x_offset, y_offset = 0.0, -20.0
            elif opt.offset_beacon == 'North_40':
                x_offset, y_offset = 0.0, -40.0
            elif opt.offset_beacon == 'South_20':
                x_offset, y_offset = 0.0, 20.0
            elif opt.offset_beacon == 'South_40':
                x_offset, y_offset = 0.0, 40.0

            # 魔卡托中心坐标 --> 魔卡托左上角右下角坐标
            # 510 = gsd * min(uav_origin_width, uav_origin_height)
            # half_dis = gsd * min(uav_origin_width, uav_origin_height)/2
            half_dis = 501/2.0  # FIXME 换数据需要重新计算
            x_mec_l = uav_mec_xy[0] - half_dis + x_offset
            x_mec_r = uav_mec_xy[0] + half_dis + x_offset
            y_mec_u = uav_mec_xy[1] - half_dis + y_offset
            y_mec_d = uav_mec_xy[1] + half_dis + y_offset

            # 魔卡托-->栅格坐标
            xpixel_11, ypixel_11 = Lonlat2Rowcol(tiff_geotrans, x_mec_l, y_mec_u)
            xpixel_22, ypixel_22 = Lonlat2Rowcol(tiff_geotrans, x_mec_r, y_mec_d)

            # 栅格行列数
            cols = abs(xpixel_22 - xpixel_11)
            rows = abs(ypixel_22 - ypixel_11)

            # tiff 保存图像选择最小，因为投影坐标大小与栅格坐标大小可能相反，存在负数情况  注意需要多个波段
            ref_img_r = tiff_r.ReadAsArray(min(xpixel_11, xpixel_22), min(ypixel_11, ypixel_22), cols, rows)
            ref_img_g = tiff_g.ReadAsArray(min(xpixel_11, xpixel_22), min(ypixel_11, ypixel_22), cols, rows)
            ref_img_b = tiff_b.ReadAsArray(min(xpixel_11, xpixel_22), min(ypixel_11, ypixel_22), cols, rows)

            ref_img = cv2.merge([ref_img_b, ref_img_g, ref_img_r])
            ref_img = cv2.resize(ref_img, (512, 512))
            sat_name = 'offset_' + opt.offset_beacon + file
            cv2.imwrite(os.path.join(ref_img_dir, sat_name), ref_img)

            # 生成sat的魔卡托中心坐标
            x_mec_c = (x_mec_l + x_mec_r) / 2.0
            y_mec_c = (y_mec_u + y_mec_d) / 2.0
            # 魔卡托坐标-->经纬度-->utm坐标
            lonlat = Mec2Lonlat(x_mec_c, y_mec_c)
            sat_utm = Lonlat2Xy(4326, 32648, lonlat[0], lonlat[1])
            # 生成reference list 保存reference.csv  mode=a+
            name_l.append(sat_name)
            eas_l.append(sat_utm[1])
            nor_l.append(sat_utm[0])

        # 保存reference.csv
        query_dict = {"easting": eas_l,
                      "northing": nor_l,
                      "name": name_l}
        pd.DataFrame(query_dict).to_csv(os.path.join(opt.save_path, "reference.csv"), mode='a', index=False)


