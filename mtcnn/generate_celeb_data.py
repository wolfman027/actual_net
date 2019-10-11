import os
import cfg
import traceback
from PIL import Image
import numpy as np
import utils

# 生成不同尺寸得人脸样本：人脸（正样本）、非人脸（负样本）、部分人脸（部分样本）
for face_size in [12, 24, 48]:
    # 样本存储路径、创建
    positive_img_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "positive")
    part_img_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "part")
    negative_img_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "negative")
    for path_dir in [positive_img_dir, part_img_dir, negative_img_dir]:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
    # 创建样本标签文件
    positive_anno_file_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "positive.txt")
    part_anno_file_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "part.txt")
    negative_anno_file_dir = os.path.join(cfg.train_gen_save_dir, str(face_size), "negative.txt")

    # 计数初始值：规范文件命名
    positive_count = 0
    part_count = 0
    negative_count = 0

    try:
        # 写入模式打开标签文件
        positive_anno_file = open(positive_anno_file_dir, "w")
        part_anno_file = open(part_anno_file_dir, "w")
        negative_anno_file = open(negative_anno_file_dir, "w")

        # 枚举列出标签文件每一行
        with open(cfg.img_label_file) as anno_label_file, open(cfg.img_face_label_file) as landmarks_label_file:
            for j, anno_line in enumerate(anno_label_file.readlines()):
                landmarks_line = landmarks_label_file.readline()
                if j < 2:
                    continue
                # 人脸框行元素转为列表
                anno_strs = anno_line.strip().split()
                # 五官坐标行元素转为列表
                landmarks_strs = landmarks_line.strip().split()
                # 取出图片名
                img_file_name = anno_strs[0]
                # 文件绝对路径
                img_file = os.path.join(cfg.img_base_dir, img_file_name)
                # 打开这个图片
                with Image.open(img_file) as img:
                    img_base_w, img_base_h = img.size  # 图片原始：宽、高
                    # 人脸框标签：左上角坐标点与框的宽高
                    base_anno_x1 = float(anno_strs[1])
                    base_anno_y1 = float(anno_strs[2])
                    base_anno_w = float(anno_strs[3])
                    base_anno_h = float(anno_strs[4])
                    # 五官点标签坐标点
                    base_lefteye_x = float(landmarks_strs[1])
                    base_lefteye_y = float(landmarks_strs[2])
                    base_righteye_x = float(landmarks_strs[3])
                    base_righteye_y = float(landmarks_strs[4])
                    base_nose_x = float(landmarks_strs[5])
                    base_nose_y = float(landmarks_strs[6])
                    base_leftmouth_x = float(landmarks_strs[7])
                    base_leftmouth_y = float(landmarks_strs[8])
                    base_rightmouth_x = float(landmarks_strs[9])
                    base_rightmouth_y = float(landmarks_strs[10])

                    # 过滤掉不符合条件的坐标
                    if min(base_anno_w, base_anno_h) < 48 \
                            or base_anno_x1 < 0 \
                            or base_anno_y1 < 0 \
                            or base_anno_w < 0 \
                            or base_anno_h < 0:
                        continue

                    # 标注不太标准：给人脸框与适当的偏移进行校准
                    align_anno_x1 = int(base_anno_x1 + base_anno_w * 0.05)
                    align_anno_y1 = int(base_anno_y1 + base_anno_h * 0.02)
                    align_anno_x2 = int(base_anno_x1 + base_anno_w * 0.95)
                    align_anno_y2 = int(base_anno_y1 + base_anno_h * 0.89)
                    align_anno_w = int(align_anno_x2 - align_anno_x1)
                    align_anno_h = int(align_anno_y2 - align_anno_y1)
                    # 左上角和右下角四个坐标点：二维的框有批次概念
                    boxes = [[align_anno_x1, align_anno_y1, align_anno_x2, align_anno_y2]]
                    # 计算出人脸中心点位置：框的中心位置
                    align_anno_cx = align_anno_x1 + align_anno_w / 2
                    align_anno_cy = align_anno_y1 + align_anno_h / 2

                    # 使正样本和部分样本数量翻倍以图片中心点随机偏移
                    for _ in range(5):
                        # 让人脸中心点有少许的偏移 得到偏移数据：宽和高，新的偏移中心点位置，五官位置
                        # 得到偏移的高和宽的长度：框的横向偏移范围：向左、向右移动了20%
                        skewing_w = np.random.randint(-align_anno_w * 0.2, align_anno_w * 0.2)
                        skewing_h = np.random.randint(-align_anno_h * 0.2, align_anno_h * 0.2)
                        # 得到偏移后的框的中心点位置
                        skewing_cx = align_anno_cx + skewing_w
                        skewing_cy = align_anno_cy + skewing_h
                        # 让人脸形成正方形（12*12，24*24,48*48），并且让坐标也有少许的偏离
                        # 边长偏移的随机数的范围；ceil大于等于该值的最小整数（向上取整）;原0.8
                        # 得到框偏移的长度：真实框高宽最小值*0.8 - 真实框高宽最小值*1.2
                        # 根据校准数据来偏移框的坐标位置
                        side_len = np.random.randint(int(min(align_anno_w, align_anno_h) * 0.8),
                                                     np.ceil(1.25 * max(align_anno_w, align_anno_h)))
                        skewing_x1 = np.max(skewing_cx - side_len / 2, 0)  # 坐标点随机偏移
                        skewing_y1 = np.max(skewing_cy - side_len / 2, 0)
                        skewing_x2 = skewing_x1 + side_len
                        skewing_y2 = skewing_y1 + side_len
                        crop_box = np.array([skewing_x1, skewing_y1, skewing_x2, skewing_y2])  # 偏移后的新框
                        # 计算坐标的偏移值
                        # 偏移量△δ=(x1-x1_)/side_len;
                        offset_x1 = (align_anno_x1 - skewing_x1) / side_len
                        offset_y1 = (align_anno_y1 - skewing_y1) / side_len
                        offset_x2 = (align_anno_x2 - skewing_x2) / side_len
                        offset_y2 = (align_anno_y2 - skewing_y2) / side_len
                        # 人的五官特征的偏移值
                        offset_px1 = (base_lefteye_x - skewing_x1) / side_len  # 人的五官特征的偏移值
                        offset_py1 = (base_lefteye_y - skewing_y1) / side_len
                        offset_px2 = (base_righteye_x - skewing_x1) / side_len
                        offset_py2 = (base_righteye_y - skewing_y1) / side_len
                        offset_px3 = (base_nose_x - skewing_x1) / side_len
                        offset_py3 = (base_nose_y - skewing_y1) / side_len
                        offset_px4 = (base_leftmouth_x - skewing_x1) / side_len
                        offset_py4 = (base_leftmouth_y - skewing_y1) / side_len
                        offset_px5 = (base_rightmouth_x - skewing_x1) / side_len
                        offset_py5 = (base_rightmouth_y - skewing_y1) / side_len
                        # 剪切下图片，并进行大小缩放
                        # “抠图”，crop剪下框出的图像
                        face_crop = img.crop(crop_box)
                        # ★按照人脸尺寸（“像素矩阵大小”）进行缩放：12/24/48；坐标没放缩
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                        # 抠出来的框和原来的框计算IOU
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        if iou > 0.6:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()  # flush：将缓存区的数据写入文件
                            face_resize.save(os.path.join(positive_img_dir, "{0}.jpg".format(positive_count)))  # 保存
                            positive_count += 1
                        elif iou > 0.4:  # 部分样本；原为0.4
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))  # 写入txt文件
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_img_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.29:  # ★这样生成的负样本很少；原为0.3
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_img_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                    # 生成负样本
                    for i in range(5):  # 数量一般和前面保持一样
                        side_len = 0
                        if (face_size == int(min(img_base_w, img_base_h) / 2)):
                            side_len = face_size
                        elif (face_size < (min(img_base_w, img_base_h) / 2)):
                            side_len = np.random.randint(face_size, min(img_base_w, img_base_h) / 2)
                        else:
                            side_len = np.random.randint(min(img_base_w, img_base_h) / 2, face_size)
                        x_ = np.random.randint(0, img_base_w - side_len)
                        y_ = np.random.randint(0, img_base_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                        if np.max(utils.iou(crop_box, np.array(boxes))) < 0.29:  # 在加IOU进行判断：保留小于0.3的那一部分；原为0.3
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_img_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
                print(j)
    except Exception as e:
        traceback.print_exc()  # 如果出现异常，把异常打印出来
    # 关闭写入文件
    finally:
        positive_anno_file.close()  # 关闭正样本txt件
        negative_anno_file.close()
        part_anno_file.close()
