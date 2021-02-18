import random
import argparse

import cv2
import numpy as np
import pycocotools.coco as coco

from config import *

def parse_args():
    parser = argparse.ArgumentParser(description='Extract positive and negative patches from a COCO-format dataset')
    parser.add_argument('--data-dir', help='The root path of a COCO-format dataset')

    args = parser.parse_args()
    return args

# RoI format: [x1, y1, x2, y2]
def check_roi(roi, src_shape, ann_list):
    def IoU(a, b):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_intersect_w = min(a[2], b[2]) - max(a[0], b[0])
        area_intersect_h = min(a[3], b[3]) - max(a[1], b[1])
        area_intersect = area_intersect_w * area_intersect_h
        if area_intersect_w <= 0 or area_intersect_h <= 0:
            return 0
        else:
            return area_intersect / (area_a + area_b - area_intersect)
    def refine_coord(x, low, high):
        if x < low:
            return low
        if x > high:
            return high
        return x

    roi[2] = refine_coord(roi[2], 1, src_shape[1])
    roi[3] = refine_coord(roi[3], 1, src_shape[0])
    roi[0] = refine_coord(roi[0], 0, roi[2] - 1)
    roi[1] = refine_coord(roi[1], 0, roi[3] - 1)

    for ann in ann_list:
        bbox = [int(x) for x in ann['bbox']]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        if IoU(roi, bbox) >= 0.75:
            return 1, roi
        if IoU(roi, bbox) >= 0.1:
            return 0, roi
    return -1, roi

def wrap_bbox(bbox, src_shape, dst_shape):
    def expand_distance(x, y, target, limit):
        mid = (x+y) // 2
        if mid < target // 2:
            return 0, target
        elif mid + target // 2 >= limit:
            return limit - target, limit
        else:
            return mid - target//2, mid - target//2 + target

    aspect_ratio = dst_shape[0] / dst_shape[1]
    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    src_w, src_h = src_shape[1], src_shape[0]

    if bbox_w / bbox_h < aspect_ratio:
        target = int(bbox_h * aspect_ratio)
        if target <= src_w:
            bbox[0], bbox[2] = expand_distance(bbox[0], bbox[2], target, src_w)
    else:
        target = int(bbox_w / aspect_ratio)
        if target <= src_h:
            bbox[1], bbox[3] = expand_distance(bbox[1], bbox[3], target, src_h)

    return bbox

def save_patch(path, src, roi):
    patch = np.zeros((roi[3] - roi[1], roi[2] - roi[0], 3), dtype=np.uint8)
    patch[0:roi[3]-roi[1], 0:roi[2]-roi[0], :] = src[roi[1]:roi[3], roi[0]:roi[2], :]
    patch = cv2.resize(patch, (patch_size[0], patch_size[1]))
    cv2.imwrite(path, patch)

if __name__ == '__main__':

    args = parse_args()

    pos_count = 0
    neg_count = 0

    coco = coco.COCO(args.data_dir + '/annotations' + '/VIS_Onshore_train.json')
    img_ids = coco.getImgIds()
    img_list = coco.loadImgs(img_ids)

    current_img_index = 0
    for img_id in img_ids:
        pos_patches = []
        neg_patches = []
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_list = coco.loadAnns(ann_ids)

        img = img_list[current_img_index]
        img = cv2.imread(args.data_dir + '/train/' + img['file_name'], cv2.IMREAD_COLOR)

        scale = scales[random.randint(0, len(scales) - 1)]
        patch_w = int(patch_size[0] * scale)
        patch_h = int(patch_size[1] * scale)

        # Generate some weak negative sample
        y = random.randint(0, img.shape[0] - patch_h - 1)
        x = random.randint(0, img.shape[1] - patch_w - 1)
        roi = [x, y, x + patch_w, y + patch_h]
        flag, roi = check_roi(roi, img.shape, ann_list)
        if flag == -1:
            save_patch(args.data_dir + '/neg' + '/weak_neg_{}.jpg'.format(neg_count), img, roi)
            neg_count = neg_count + 1

        for ann in ann_list:

            bbox = [int(x) for x in ann['bbox']]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            x = bbox[0]
            y = bbox[1]

            # Generate a strong negative sample
            for trial in range(20):
                delta_y = random.randint(patch_h//3, patch_h)
                delta_x = random.randint(patch_w//3, patch_w)
                if random.randint(0, 1):
                    delta_x = -delta_x
                if random.randint(0, 1):
                    delta_y = -delta_y
                
                roi = [x + delta_x, y + delta_y, x + delta_x + patch_w, y + delta_y + patch_h]
                flag, roi = check_roi(roi, img.shape, ann_list)
                if flag == -1:
                    save_patch(args.data_dir + '/neg' + '/strong_neg_{}.jpg'.format(neg_count), img, roi)
                    neg_count = neg_count + 1
                    break

            # Generate positive samples
            roi = wrap_bbox(bbox, img.shape, patch_size)
            _, roi = check_roi(roi, img.shape, ann_list)
            save_patch(args.data_dir + '/pos' + '/pos_{}.jpg'.format(pos_count), img, roi)
            pos_count = pos_count + 1
            
            for trial in range(100):
                delta_y = random.randint(-patch_h//8, patch_h//8)
                delta_x = random.randint(-patch_w//8, patch_w//8)
                roi = [x + delta_x, y + delta_y, x + delta_x + patch_w, y + delta_y + patch_h]
                flag, roi = check_roi(roi, img.shape, ann_list)
                if flag == 1:
                    save_patch(args.data_dir + '/pos' + '/pos_{}.jpg'.format(pos_count), img, roi)
                    pos_count = pos_count + 1
                    break

        current_img_index = current_img_index + 1

    print(pos_count, neg_count)