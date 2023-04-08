#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/8 18:41
# @File  : t1.py
# @Author:
# @Desc  : fake msls data
import os
import numpy as np

def generate_fake_images():
    msls_path = "datasets/msls_val"
    dbimages_file = os.path.join(msls_path, "msls_val_dbImages.npy")
    dbimages = np.load(dbimages_file)
    qimages_file = os.path.join(msls_path, "msls_val_qImages.npy")
    qimages = np.load(qimages_file)
    source_db_image_file = os.path.join(msls_path, "train_val/melbourne/database/images/_Bkgic7JtCaaMGrgw6kukQ.jpg")
    source_q_image_file = os.path.join(msls_path, "train_val/melbourne/query/images/yOCOTojshHsfJd0g5B7jPP.jpg")
    # db文件夹
    print(f"正在复制{len(dbimages)}个数据库文件到{msls_path}目录下")
    for i in range(len(dbimages)):
        target_db_image_file = os.path.join(msls_path, dbimages[i])
        # 如果目标文件夹不存在，则创建
        if not os.path.exists(os.path.dirname(target_db_image_file)):
            os.makedirs(os.path.dirname(target_db_image_file))
        os.system(f"cp {source_db_image_file} {target_db_image_file}")
    # 查询文件夹创建
    print(f"正在复制{len(qimages)}个查询文件到{msls_path}目录下")
    for i in range(len(qimages)):
        target_q_image_file = os.path.join(msls_path, qimages[i])
        # 如果目标文件夹不存在，则创建
        if not os.path.exists(os.path.dirname(target_q_image_file)):
            os.makedirs(os.path.dirname(target_q_image_file))
        os.system(f"cp {source_q_image_file} {target_q_image_file}")

if __name__ == '__main__':
    generate_fake_images()
