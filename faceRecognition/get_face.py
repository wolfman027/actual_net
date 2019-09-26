import os
import cv2
import cfg

face_cascade = cv2.CascadeClassifier(os.path.join(cfg.net_main_dir, 'haarcascade_frontalface_default.xml'))

for person_dir in os.listdir(cfg.base_img_dir):
    label = int(person_dir)
    label_dir = os.path.join(cfg.train_face_img_main_dir, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    for pic_filename in os.listdir(os.path.join(cfg.base_img_dir, person_dir)):
        img = cv2.imread(os.path.join(cfg.base_img_dir, person_dir, pic_filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(boxes) == 1:
            x, y, w, h = boxes[0]
            corp = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(cfg.train_face_img_main_dir, label_dir, pic_filename), corp)







