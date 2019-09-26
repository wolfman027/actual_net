# coco 数据集转换
# 保存训练图片的标签
# images/1.jpg 1(物体类别) 12(中心点cx) 13(中心点cy) 51(宽) 18(高) 2 22 31 55 98 2 44 33 62 62

from pycocotools.coco import COCO
import cfg

with open(cfg.label_file, "w+") as f:
    # 载入训练集json数据
    coco = COCO(cfg.coco_ann_file)
    # 获取我们感兴趣得类别
    catIds = coco.getCatIds(catNms=cfg.coco_class)
    # 获得COCO数据集的所有分类id'
    cats = {}
    for cat in coco.loadCats(catIds):
        cats[cat['id']] = cat['name']

    # 获取指定分类ID下的所有图片ID
    imgIds = []
    for catId in catIds:
        cat_imgIds = coco.getImgIds(catIds=catId)
        imgIds.extend(cat_imgIds)
    imgIds = set(imgIds)
    imgIds = list(imgIds)

    # 循环图片id，获取图片的文件名以及标签数据并保存到文件中
    i = 0
    for imgId in imgIds:
        # 获取图片信息
        # {'license': 1, 'file_name': '000000225848.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000225848.jpg',
        # 'height': 480, 'width': 640, 'date_captured': '2013-11-16 01:32:39',
        # 'flickr_url': 'http://farm2.staticflickr.com/1035/674645314_995cb4b793_z.jpg', 'id': 225848}
        img = coco.loadImgs(imgId)[0]
        # 获取图片的名称
        img_file_name = img['file_name']
        # 获取图片的宽和高
        w, h = img['width'], img['height']
        # 图片在416像素中，图片的宽高缩放比例
        w_scale, h_scale = w/cfg.img_width, h/cfg.img_height

        # 获取图片所有框信息
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        f.write(img_file_name)
        for ann in anns:
            categoryId = ann['category_id']
            categoryName = cats[categoryId]
            cls = cfg.coco_class.index(categoryName)

            # 由于框的中心坐标为框的左上角坐标，所以需要计算中心点坐标
            # 中心点坐标 = 原点坐标 + 一半宽与一半高
            _x1, _y1, _w, _h = ann['bbox']
            _w0_5, _h0_5 = _w/2, _h/2
            _cx, _cy = _x1+_w0_5, _y1+_h0_5
            x1, y1, w, h = int(_cx/w_scale), int(_cy/h_scale), int(_w/w_scale), int(_h/h_scale)
            # 类别，中心点x，中心点y，宽，高
            f.write(" {} {} {} {} {}".format(cls, x1, y1, w, h))
        f.write("\n")
        f.flush()
