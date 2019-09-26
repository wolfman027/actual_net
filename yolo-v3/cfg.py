# 配置相关信息

# coco 训练集json数据
coco_ann_file = r'D:\ai\ai\train_large_file\coco\instances_train2017.json'

# coco 训练集标签文件
label_file = r'data/coco_label.txt'

img_base_dir = r'D:\ai\ai\train2017\train2017'

# iou threshold
iou_threshold = 0.5

# img训练得高和宽
img_width = 416
img_height = 416

# coco 训练集所有分类
coco_class = ["person",
              "bicycle",
              "car",
              "motorcycle",
              "airplane",
              "bus",
              "train",
              "truck",
              "boat",
              "traffic light",
              "fire hydrant",
              "stop sign",
              "parking meter",
              "bench",
              "bird",
              "cat",
              "dog",
              "horse",
              "sheep",
              "cow",
              "elephant",
              "bear",
              "zebra",
              "giraffe",
              "backpack",
              "umbrella",
              "handbag",
              "tie",
              "suitcase",
              "frisbee",
              "skis",
              "snowboard",
              "sports ball",
              "kite",
              "baseball bat",
              "baseball glove",
              "skateboard",
              "surfboard",
              "tennis racket",
              "bottle",
              "wine glass",
              "cup",
              "fork",
              "knife",
              "spoon",
              "bowl",
              "banana",
              "apple",
              "sandwich",
              "orange",
              "broccoli",
              "carrot",
              "hot dog",
              "pizza",
              "donut",
              "cake",
              "chair",
              "couch",
              "potted plant",
              "bed",
              "dining table",
              "toilet",
              "tv",
              "laptop",
              "mouse",
              "remote",
              "keyboard",
              "cell phone",
              "microwave",
              "oven",
              "toaster",
              "sink",
              "refrigerator",
              "book",
              "clock",
              "vase",
              "scissors",
              "teddy bear",
              "hair drier",
              "toothbrush"]

# 类别数量
class_num = len(coco_class)

# 建议框组合
anchors_group = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

# 建议框区域面积
anchors_group_area = {
    13: [x*y for x, y in anchors_group[13]],
    26: [x*y for x, y in anchors_group[26]],
    52: [x*y for x, y in anchors_group[52]]
}


if __name__ == '__main__':
    print(coco_class[0])
    print(coco_class[24])
    print(coco_class[56])
    print(coco_class[65])























