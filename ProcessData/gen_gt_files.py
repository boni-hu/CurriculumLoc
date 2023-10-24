import numpy as np
import csv
from os.path import join
import pandas as pd

# data_q = []
# data_db = []
#
path = '/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/'
#
# with open(join(path, 'reference.csv'), 'r') as csv_file_db:
#     csv_data = csv.DictReader(csv_file_db)
#     for row in csv_data:
#         data_db.append(row)
#
# with open(join(path, 'query.csv'), 'r') as csv_file_q:
#     csv_data_q = csv.DictReader(csv_file_q)
#     for row in csv_data_q:
#         data_q.append(row)
#
# numQ = len(data_q)
# numDb = len(data_db)
#
# utmDb = [[float(f["easting"]), float(f["northing"])] for f in data_db]
# utmQ = [[float(f["easting"]), float(f["northing"])] for f in data_q]
posDistThr = 50
qData = pd.read_csv(join(path, 'query.csv'))
qData = qData.sort_values(by='name', ascending=True)
dbData = pd.read_csv(join(path, 'reference.csv'))
dbData = dbData.sort_values(by='name', ascending=True)
utmQ = qData[['easting', 'northing']].values.reshape(-1, 2)
utmDb = dbData[['easting', 'northing']].values.reshape(-1, 2)

np.savez('../patchnetvlad/dataset_gt_files/mavic_npu_dist50', utmQ=utmQ, utmDb=utmDb, posDistThr=posDistThr)

print("Save Done!")
