'''
Set the config variable.
'''
import os, sys

from configparser import RawConfigParser
import json

config_paths = [
    './config.cfg',
    './data/config/config.cfg',
    './HOG-SVM-python/data/config/config.cfg'
]
config = RawConfigParser()
for path in config_paths:
    if os.path.exists(path):
        config.read(path)

min_wdw_sz = tuple(json.loads(config.get("hog","min_wdw_sz")))
step_size = tuple(json.loads(config.get("hog", "step_size")))
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
transform_sqrt = config.getboolean("hog", "transform_sqrt")

pos_feat_ph = config.get("paths", "pos_feat_ph")
neg_feat_ph = config.get("paths", "neg_feat_ph")
model_path = config.get("paths", "model_path")

threshold = config.getfloat("nms", "threshold")

scales = json.loads(config.get("preprocess", "scales"))
patch_size = json.loads(config.get("preprocess", "patch_size"))
neg_samples = config.getint("preprocess", "neg_samples")
pos_samples = config.getint("preprocess", "pos_samples")
