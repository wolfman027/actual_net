import os
import cfg
import SimpleITK as sitk
import cv2
from PIL import Image


def dcmtopng(filename, outpath, data):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    for img_item in img_array:
        cv2.imwrite("%s/%s.png" % (outpath,data.split('.')[0]), img_item)


pationts = os.listdir(cfg.base_data_dir)
count1=0
count2=0
for pationt in pationts:
    dirs = os.listdir(os.path.join(cfg.base_data_dir, pationt))
    for dir in dirs:
        datasets = os.listdir(os.path.join(os.path.join(cfg.base_data_dir, pationt), dir))
        for data in datasets:
            file_path = os.path.join(os.path.join(os.path.join(cfg.base_data_dir, pationt), dir), data)
            if data.split('.')[1] == 'dcm':
                dcmtopng(file_path, os.path.join(cfg.train_data_dir, 'data'), str(count1)+'.png')
                count1 += 1
            elif data.split('.')[1] == 'png':
                image = Image.open(file_path)
                image.save(os.path.join(os.path.join(cfg.train_data_dir, 'label'), str(count2) + '.png'))
                count2 += 1
