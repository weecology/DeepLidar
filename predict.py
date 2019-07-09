import keras
import pyfor
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

#parse args
import argparse
import glob

#DeepForest
from DeepForest.evalmAP import _get_detections

#Set training or training
mode_parser     = argparse.ArgumentParser(description='Prediction of a new image')
mode_parser.add_argument('--model', help='path to training model' )
mode_parser.add_argument('--image', help='image or directory of images to predict' )
mode_parser.add_argument('--output_dir', default="snapshots/images/")

args=mode_parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

#load config
DeepForest_config = config.load_config()
# adjust this to point to your downloaded/trained model

# load retinanet model
model = models.load_model(args.model, backbone_name='resnet50', convert=True, nms_threshold=args.nms_threshold)
labels_to_names = {0: 'Tree'}


