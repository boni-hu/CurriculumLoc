import os
from os.path import join
from os.path import exists
import numpy as np
import matchs
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import pandas as pd
import time
import argparse
import cv2
import configparser

import warnings
warnings.filterwarnings("ignore", category=Warning)

def parse_gt_file(gtfile):
    print('Parsing ground truth data file...')
    gtdata = np.load(gtfile)
    return gtdata['utmQ'], gtdata['utmDb'], gtdata['posDistThr']

def get_positives(utmQ, utmDb, posDistThr):
    # positives for evaluation are those within trivial threshold range
    # fit NN to find them, search by radius
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utmDb)
    distances, positives = knn.radius_neighbors(utmQ, radius=posDistThr)

    return positives

def compute_recall(query_idx_list, gt, predictions, numQ, n_values, recall_str=''):
    start2 = time.perf_counter()
    correct_at_n = np.zeros(len(n_values))

    for i, pred in enumerate(predictions):
        for j, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[query_idx_list[i]])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall {}@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))

    recall_time = time.perf_counter()
    print('Compute recall time is %6.3f' % (recall_time - start2))

    return all_recalls

def write_recalls(opt, netvlad_recalls, cnnmatch_recalls, n_values, rec_file):
    with open(rec_file, 'w') as res_out:
        res_out.write(str(opt)+'\n')
        res_out.write("n_values: "+str(n_values)+'\n')
        for n in n_values:
            res_out.write("Recall {}@{}: {:.4f}\n".format('netvlad_match', n, netvlad_recalls[n]))
            res_out.write("Recall {}@{}: {:.4f}\n".format('cnn_match', n, cnnmatch_recalls[n]))

def image_point2gps_point(keypoint_position: np.array, image_position: np.array, gsd, img_size):
    half_img_size = img_size/2
    point_position = np.array([image_position[0]+(keypoint_position[0]-half_img_size)*gsd, image_position[1]+(half_img_size - keypoint_position[1])*gsd])
    alt = np.zeros(keypoint_position.shape[1])
    point_position= np.vstack([*point_position,alt])
    return point_position



def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Match')
    parser.add_argument('--config_path', type=str, default='performance.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for cnn-matching')
    parser.add_argument('--ground_truth_path', type=str, default='/home/a409/users/huboni/Projects/code/Patch-NetVLAD/patchnetvlad/dataset_gt_files/mavic-xjtu_dist50.npz',
                        help='ground truth file dist 50')
    parser.add_argument('--netvlad_result_path', type=str, default='/home/a409/users/huboni/Projects/code/Patch-NetVLAD/patchnetvlad/results/mavic-xjtu_2048_7_dist50/NetVLAD_predictions.txt',
                        help='netvlad predictions result path')
    parser.add_argument('--reference_origin_path', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic-xjtu/reference.csv',
                        help='reference csv path')
    parser.add_argument('--query_origin_path', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic-xjtu/query.csv',
                        help='query csv path')
    parser.add_argument('--out_dir', type=str, default='/home/a409/users/huboni/Projects/code/cnn-matching/result/mavic-xjtu/',
                        help='Dir to save recall')
    parser.add_argument('--out_1019_dir', type=str, default='/home/a409/users/huboni/Projects/code/cnn-matching/result/mavic-xjtu/',
                        help='Dir to save recall')
    parser.add_argument('--out_file', type=str, default='cnn_mavic-xjtu_dist50_v50',
                        help='Dir to save recall')
    parser.add_argument('--fig_res_file', type=str, default='pnp_fig_res_npu_dist20.csv')
    parser.add_argument('--global_res_file', type=str, default='pnp_fig_res_npu_dist20.csv')
    parser.add_argument('--rerank_res_file', type=str, default='pnp_fig_res_npu_dist20.csv')
    parser.add_argument('--rerank1_res_file', type=str, default='pnp_fig_res_npu_dist20.csv')
    parser.add_argument('--pinpoint_res_file', type=str, default='pnp_fig_res_npu_dist20.csv')


    parser.add_argument('--model_type', type=str, default='cnn', help='or cnn')
    parser.add_argument('--img_size', type=int, default=500, help='or 224')
    parser.add_argument('--checkpoint', type=str, default='/home/a409/users/huboni/Projects/code/cnn-matching/models/d2_tf.pth',
                        help='or ./models/d2_tf.pth')

    opt = parser.parse_args()
    print(opt)

    n_values = [1, 5, 10, 20, 50]  # FIXME
    if opt.img_size == 500:
        # gsd = 300 / 500 # alto
        # gsd = 300 / 500 # xjtu
        gsd = 0.167 # npu
        print("gsd:", gsd)
        # k = np.array([[345, 0, 247.328], [0, 345, 245], [0, 0, 1]]) # ALTO
        # k = np.array([[350, 0, 250], [0, 350, 250], [0, 0, 1]]) # xjtu
        k = np.array([[475.48, 0, 250], [0, 475.48, 250], [0, 0, 1]]) # npu
        # k = np.array([[3051.6 / (3000/500), 0, 1500/(3000/500)], [0, 3051.6/(3000/500), 1500/(3000/500)], [0, 0, 1]])  # TerraTrack
    elif opt.img_size == 224:
        gsd = 300 / 500 * (500/224)
        print("gsd:", gsd)
        k = np.array([[345 / (500/224), 0, 247.328 / (500/224)], [0, 345 / (500/224), 245 / (500/224)], [0, 0, 1]])
    num_rec = (n_values[-1])
    with open(opt.netvlad_result_path, 'r') as file:
        lines = file.readlines()[2:]  # 前两行是注释数据
    numQ = len(lines)//num_rec
    print("numQ:", numQ)

    query_idx_list_all = []
    refidx_list = []
    refidx_inliners_list = []

    start0 = time.perf_counter()
    # rerank by cnn match + flann + ransac
    qData = pd.read_csv(opt.query_origin_path)
    # qData = qData.sort_values(by='name', ascending=True)
    print("qData", qData)
    dbData = pd.read_csv(opt.reference_origin_path)
    # dbData = dbData.sort_values(by='name', ascending=True)
    # print("daData", dbData)
    for i in tqdm(range(len(lines)), desc='cnn_match_ing'):

        query_file = lines[i].split(',')[0].strip() # query_file 按照idx的顺序逐个存取的，所以只需要将ref的idx关联起来
        query_name = os.path.basename(query_file)
        reference_file = lines[i].split(',')[1].strip() # reference_name
        # reference_name = os.path.join(reference_file.split('/')[-2], reference_file.split('/')[-1]) # alto
        reference_name = os.path.basename(reference_file) # terratrack FIXME
        # print("reference_name", reference_name)
        query_idx = qData[qData.name == query_name].index.to_list()[0]
        query_idx_list_all.append(query_idx)
        ref_idx = dbData[dbData.name == reference_name].index.to_list()[0]
        inliners, avg_dist, q_kps, r_kps = matchs.cnn_match(query_file, reference_file, opt.model_type, opt.checkpoint, opt.img_size)
        # print("-------current avg_list------:", avg_dist)
        # print("--------num inliners---------:", inliners)
        refidx_list.append(ref_idx)
        refidx_inliners_list.append([ref_idx, avg_dist, q_kps, r_kps])
    query_idx_list = query_idx_list_all[0: len(query_idx_list_all) : num_rec]
    print("-----------q idx list:", query_idx_list)
    match_time = time.perf_counter()
    print('CNN match time is %6.3f' % (match_time - start0))

    # gen origin input netvlad predictions
    refidx_list_split = [refidx_list[i:i + num_rec] for i in range(0, len(refidx_list), num_rec)]
    netvlad_predictions = np.array(refidx_list_split)

    # split every query rematch[ref_idx inliners]
    refidx_inliners_list_split =[refidx_inliners_list[i:i+num_rec] for i in range(0, len(refidx_inliners_list), num_rec)]
    # reranking every query rematch[ref_idx inliners] by inliners avg_dist
    start1 = time.perf_counter()
    cnnmatch_predictions = []
    pinpoint_utms = []
    name_l = []
    arr_ppl = []
    arr_r1l = []
    global_candidates_utm = []
    rerank_candidates_utm = []
    rerank_1_utm = []
    pinpoint_utm_list = []
    pass_count = 0
    fig_csv_file = os.path.join(opt.out_dir, opt.fig_res_file)
    global_csv_file = os.path.join(opt.out_1019_dir, opt.global_res_file)
    rerank_csv_file = os.path.join(opt.out_1019_dir, opt.rerank_res_file)
    rerank1_csv_file = os.path.join(opt.out_1019_dir, opt.rerank1_res_file)
    pinpoint_csv_file = os.path.join(opt.out_1019_dir, opt.pinpoint_res_file)



    pnp_select_res_dict = pd.DataFrame()
    for i, idx_inliners_list in enumerate(refidx_inliners_list_split):
        print("-----------------",i)
        reranked_rf_idx = np.array(sorted(idx_inliners_list, key=lambda x:x[1], reverse=False))[:, 0][:5] # 按照第二位 降序排列 并输出第一位 FIXME Terratrack
        cnnmatch_predictions.append(reranked_rf_idx)
        reranked_query_kps = np.array(sorted(idx_inliners_list, key=lambda x:x[1], reverse=False))[:, 2][:5] # FIXME
        reranked_ref_kps = np.array(sorted(idx_inliners_list, key=lambda x:x[1], reverse=False))[:, 3][:5] # FIXME
        query_utm = np.array(qData.loc[query_idx_list[i], ["easting", "northing"]])
        query_utm = query_utm[::-1] # terratrack swap easting northing
        print("query name:", qData.loc[query_idx_list[i], ["name"]])

        # print("query name:", qData.loc[query_id, ["name"]])
        ref_image_position_o = dbData.loc[reranked_rf_idx.tolist(), ["easting", "northing"]]
        # ref_image_position = np.array([ref_image_position_o["easting"], ref_image_position_o["northing"]]).transpose() # alto
        ref_image_position = np.array([ref_image_position_o["northing"], ref_image_position_o["easting"]]).transpose() # terratrack

        ref_kpts_position = []
        for j, image_position in enumerate(ref_image_position):
            ref_kpts_position.append(image_point2gps_point(np.array(reranked_ref_kps[j]).transpose(), image_position, gsd, opt.img_size))
        ref_kpts_position = np.hstack(ref_kpts_position).transpose() # 关键点GPS坐标
        query_kps = [np.array(ii) for ii in reranked_query_kps]
        query_kps_position = np.vstack(query_kps)

        success, R_vec, t, inliers = cv2.solvePnPRansac(ref_kpts_position, query_kps_position, k, np.zeros(4),
                                                       flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=5000,
                                                       reprojectionError=10)
        # print("R_vec:", R_vec)
        print("success:", success)
        if not success:
            continue
        r_w2c, _ = cv2.Rodrigues(R_vec)
        t_w2c = t
        r_c2w = np.linalg.inv(r_w2c)
        t_c2w = -r_c2w @ t_w2c
        # print("---------t_c2w-------:", t_c2w)
        pinpoint_utm = t_c2w[:, 0][:2]
        name_l.append(qData.loc[query_idx_list[i], ["name"]])
        pinpoint_utms.append(pinpoint_utm)
        # print("pinpoint utm:", pinpoint_utm)
        pp_loss = np.linalg.norm(query_utm - pinpoint_utm)
        print("pp_loss", pp_loss)
        # if pp_loss>10:
        #     pass_count += 1
        #     continue
        arr_ppl.append(pp_loss)
        r1_utm = dbData.loc[reranked_rf_idx.tolist()[0], ["easting", "northing"]]
        print("r1 name:", dbData.loc[reranked_rf_idx.tolist()[0], ["name"]])
        r2_utm = dbData.loc[reranked_rf_idx.tolist()[1], ["easting", "northing"]]
        r3_utm = dbData.loc[reranked_rf_idx.tolist()[2], ["easting", "northing"]]
        r4_utm = dbData.loc[reranked_rf_idx.tolist()[3], ["easting", "northing"]]
        r5_utm = dbData.loc[reranked_rf_idx.tolist()[4], ["easting", "northing"]]
        # swap easting northing below
        r1_loss = np.linalg.norm(query_utm - np.array(r1_utm)[::-1])
        r2_loss = np.linalg.norm(query_utm - np.array(r2_utm)[::-1])
        r3_loss = np.linalg.norm(query_utm - np.array(r3_utm)[::-1])
        r4_loss = np.linalg.norm(query_utm - np.array(r4_utm)[::-1])
        r5_loss = np.linalg.norm(query_utm - np.array(r5_utm)[::-1])

        r_mean_loss = sum([r1_loss,r2_loss,r3_loss,r4_loss,r5_loss])/5
        arr_r1l.append(r1_loss)
        print("localization loss:", pp_loss)
        print("recall@1 loss:", r1_loss)
        # save csv for localization fig
        # if r_mean_loss-pp_loss>=50.0 and pp_loss<2.0:
        # if r1_loss-pp_loss>10.0:
        if True:
            global_candidates_utm.extend((dbData.loc[np.array(idx_inliners_list)[:,0][:5], ["easting", "northing"]]).to_numpy().tolist())
            rerank_candidates_utm.extend((dbData.loc[reranked_rf_idx.tolist()[:5], ["easting", "northing"]]).to_numpy().tolist())
            rerank_1_utm.append((dbData.loc[reranked_rf_idx.tolist()[0], ["easting", "northing"]]).to_numpy().tolist())
            pinpoint_utm_list.append(pinpoint_utm.tolist())

            fig_res = dict(query_name = (qData.loc[query_idx_list[i], ["name"]]).to_numpy(),
                                   global_recall5_idx = (np.array(idx_inliners_list)[:, 0][:5]), global_recall5_name = (dbData.loc[np.array(idx_inliners_list)[:, 0][:5],["name"]]).to_numpy(),
                                   global_recall5_utm =  (dbData.loc[np.array(idx_inliners_list)[:,0][:5], ["easting", "northing"]]).to_numpy(),
                                   rerank_recall5_idx = (reranked_rf_idx.tolist()[:5]), rerank_recall5_name = (dbData.loc[reranked_rf_idx.tolist()[:5], ["name"]]).to_numpy(),
                                   rerank_recall5_utm = (dbData.loc[reranked_rf_idx.tolist()[:5], ["easting", "northing"]]).to_numpy(),
                                   rerank_recall1_idx =  (reranked_rf_idx.tolist()[0]), rerank_recall1_name = (dbData.loc[reranked_rf_idx.tolist()[:1], ["name"]]).to_numpy(),
                                   rerank_recall1_utm = (dbData.loc[reranked_rf_idx.tolist()[0], ["easting", "northing"]]).to_numpy(),
                                   pinpoint_utm = pinpoint_utm,
                                   query_utm = query_utm, r1_loss = r1_loss, pp_loss=pp_loss)
            # print(fig_res)
            pnp_select_res_dict = pnp_select_res_dict._append(fig_res, ignore_index=True)

    pd.DataFrame(pnp_select_res_dict).to_csv(fig_csv_file, index=True, encoding='gbk',float_format='%.6f')
    pd.DataFrame(global_candidates_utm).to_csv(global_csv_file, index=True, encoding='gbk',float_format='%.6f')
    pd.DataFrame(rerank_candidates_utm).to_csv(rerank_csv_file, index=True, encoding='gbk',float_format='%.6f')
    pd.DataFrame(rerank_1_utm).to_csv(rerank1_csv_file, index=True, encoding='gbk',float_format='%.6f')
    pd.DataFrame(pinpoint_utm_list).to_csv(pinpoint_csv_file, index=True, encoding='gbk',float_format='%.6f')



    # print("global candidates list:", global_candidates_utm)
    # print("rerank candidates list:", rerank_candidates_utm)
    # print("rerank 1 utm list:", rerank_1_utm)
    # print("pinpoint utm list", pinpoint_utm_list)

    print("pass count:", pass_count)
    print("avg pinpoint loss:", np.mean(np.array(arr_ppl)))
    print("avg recall@1 loss:", np.mean(np.array(arr_r1l)))
    print("var pinpoint loss:", np.var(np.array(arr_ppl)))
    print("var recall@1 loss:", np.var(np.array(arr_r1l)))
    print("std pinpoint loss:", np.std(np.array(arr_ppl)))
    print("std recall@1 loss:", np.std(np.array(arr_r1l)))
    # 保存pinpoint.csv
    pinpoint_dict = {"easting":np.array(pinpoint_utms)[:,0],
                  "northing":np.array(pinpoint_utms)[:,1],
                  "name":name_l}

    out_csv_file = os.path.join(opt.out_file + '.csv')
    print("pinpoint utm save to %s" % out_csv_file)
    pd.DataFrame(pinpoint_dict).to_csv(os.path.join(opt.out_dir, 'pinpoint_utm', out_csv_file), index=False)


    cnnmatch_predictions = np.array(cnnmatch_predictions)
    cnn_pred_time = time.perf_counter()
    print('rerank match get cnn predict time is %6.3f' % (cnn_pred_time - start1))

    # get ground truth
    utmQ,utmDb,posDistThr = parse_gt_file(opt.ground_truth_path)
    gt = get_positives(utmQ, utmDb, posDistThr)

    netvlad_recalls = compute_recall(query_idx_list, gt, netvlad_predictions, numQ, n_values, 'netvlad_match')
    cnnmatch_recalls = compute_recall(query_idx_list, gt, cnnmatch_predictions, numQ, n_values, 'cnn_match')

    # out_recall_file = os.path.join(opt.out_dir, 'recalls', opt.checkpoint.split('/')[-3] + '_' + str(opt.img_size) + '.txt')
    out_recall_file = os.path.join(opt.out_dir, 'recalls', opt.out_file + '.txt')

    print('Writing recalls to', out_recall_file)
    write_recalls(opt, netvlad_recalls, cnnmatch_recalls, n_values, out_recall_file)




if __name__ == "__main__":
    main()

