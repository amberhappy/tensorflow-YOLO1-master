import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import utils.config as Config
#np.set_printoptions(threshold=np.inf)

##################test of resize image
flipped = 'True'
image_size=448

impath= "./data/VOCdevkit/VOC2012/JPEGImages"
imname = '2012_004005.jpg'
imname = os.path.join(impath,imname)
image = cv2.imread(imname)
image = cv2.resize(image, (image_size, image_size))
#print(image.shape)
#cv2.imshow("ori",image)
if flipped:
    image = image[:, ::-1, :]
#image = np.asarray(image,dtype='uint8')
#print(image.shape)
#cv2.imshow("flip",image)
#cv2.waitKey(0)

####################test of load_labels   载入数据
train_percentage=0.9
data_path = './data/VOCdevkit/VOC2012/JPEGImages/'
img_index = os.listdir(data_path)
img_index = [i.replace('.jpg','') for i in img_index]
#print(img_index)
import random
random.shuffle(img_index)
train = int(len(img_index) * (1 -train_percentage))
image_train_index = img_index[train:]
image_val_index = img_index[:train]

get_labels_train = []
get_labels_val = []


###############test of load_pascal_annotation  提取坐标  提取注释信息
cell_size = 7
imname = os.path.join("./data/VOCdevkit/VOC2012/JPEGImages",  '2007_000323.jpg')
im = cv2.imread(imname)
h_ratio = 1.0 * image_size / im.shape[0]
w_ratio = 1.0 * image_size / im.shape[1]
image = cv2.resize(im, (image_size, image_size))

label = np.zeros((cell_size, cell_size, 25))
filename = os.path.join("./data/VOCdevkit/VOC2012/Annotations",  '2007_000323.xml')
tree = ET.parse(filename)
# 所有目标
objs = tree.findall('object')
classes_to_id = Config.classes_dict
for obj in objs:
    # 坐标
    bbox = obj.find('bndbox')
    # 将坐标变换到image_size尺寸
    x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, image_size - 1), 0)
    y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, image_size - 1), 0)
    x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, image_size - 1), 0)
    y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, image_size - 1), 0)
    # 目标类别
    class_id = classes_to_id[obj.find('name').text.lower().strip()]
    # 坐标转换成中心点形式
    boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
    # 按ceil分坐标
    x_id = int(boxes[0] * cell_size / image_size)
    y_id = int(boxes[1] * cell_size / image_size)
    if label[y_id, x_id, 0] == 1:
        continue
    label[y_id, x_id, 0] = 1
    label[y_id, x_id, 1:5] = boxes
    label[y_id, x_id, 5 + class_id] = 1

    print(x_id,y_id)
    print(label)

#     cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow("rec",image)
# cv2.waitKey(0)


