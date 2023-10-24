import os
import argparse


parser = argparse.ArgumentParser(description='origin data path')
parser.add_argument('--data_path', type=str, default='/home/a409/users/huboni/Projects/dataset/TerraTrack/mavic_npu/reference_images_500', help='origin images path')
parser.add_argument('--save_path', type=str, default='../patchnetvlad/dataset_imagenames/mavic_npu_reference_imageNames_index.txt',
                    help='save images txt path')
          
    
opt = parser.parse_args()
print(opt)


for dir in os.listdir(opt.data_path):
    sub_data_path = os.path.join(opt.data_path, dir)
    if os.path.isdir(sub_data_path):
        for f in os.listdir(sub_data_path):
            image_name = os.path.join(sub_data_path, f)
            image_name_w = os.path.join(*(image_name.split('/')[-3:]))
            with open(opt.save_path, 'a') as txt:
                txt.write(image_name_w)
                txt.write('\n')
    else:
        image_name = os.path.join(opt.data_path, dir)
        # image_name_w = os.path.join(*(image_name.split('/')[-2:]))
        with open(opt.save_path, 'a') as txt:
            # txt.write(image_name_w)
            txt.write(image_name)
            txt.write('\n')
