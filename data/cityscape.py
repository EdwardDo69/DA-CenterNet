import cv2
import numpy as np

import os
import json


CLASSES = ('person', 'rider', 'car', 'truck', 
           'bus', 'train', 'motorcycle', 'bicycle')

IMAGE_SET = {'train': ['train'], 'test': ['val']}
           

class CityScapeDetection(object):
    
    def __init__(self,
                 root,
                 image_set='train'):
    
        self.root = root
        self.images_path = []
        self.labels_path = []
        self.labels = []
        
        for mode in IMAGE_SET[image_set]:
            images_root = os.path.join(self.root, 'images', mode)
            labels_root = os.path.join(self.root, 'labels', mode)
            
            images_name = os.listdir(images_root)
            
            for img in images_name:
                lb = img.split('_')[:3] + ['gtFine', 'polygons.json']
                lb = '_'.join(lb)
                
                img_path = os.path.join(images_root, img)
                lb_path = os.path.join(labels_root, lb)
                label = self.read_json(lb_path)
                
                if len(label) == 0:
                    continue
                
                self.images_path.append(img_path)
                self.labels_path.append(lb_path)
                self.labels.append(label)
                
        assert len(self.images_path) == len(self.labels_path)
        assert len(self.labels_path) == len(self.labels)
        
    def read_json(self, label_path):
        label = []
        
        with open(label_path, 'r') as f:
            root = json.load(f)
            
        img_w = root['imgWidth']
        img_h = root['imgHeight']
        
        objs = root['objects']
        
        for obj in objs:
            class_name = obj['label'].lower().strip()
            
            if class_name not in CLASSES:
                continue
                
            class_idx = CLASSES.index(class_name)
            
            polygon = np.array(obj['polygon'])
            xmin = min(polygon[:, 0])
            ymin = min(polygon[:, 1])
            xmax = max(polygon[:, 0])
            ymax = max(polygon[:, 1])
            
            cx = ((xmax + xmin) / 2.) / img_w
            cy = ((ymax + ymin) / 2.) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            label.append([class_idx, cx, cy, w, h])
            
        label = np.array(label).reshape(-1, 5)
        label[:, 1:] = np.clip(label[:, 1:], a_min=0., a_max=1.)
        return label
        
    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx], cv2.IMREAD_COLOR)
        label = self.labels[idx].copy()
        return img, label
        
    def __len__(self):
        return len(self.images_path)