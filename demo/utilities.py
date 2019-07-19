#utility functions for demo
import os
import yaml
import sys
from datetime import datetime
import keras
import tensorflow as tf
import cv2
import numpy as np
import copy

#DeepForest
sys.path.append("../")
from keras_retinanet import models
from keras_retinanet.utils import image as keras_retinanet_image
from keras_retinanet.utils.visualization import draw_detections
from DeepForest.utils import generators
from DeepForest.utils import image_utils
from DeepForest import config

def label_to_name(label):
        """ Map label to name.
        """
        return "Tree"

def read_config():
        with open("_config.yml", 'r') as f:
                config = yaml.load(f)
        return config

def read_model(model_path, config):
        model = models.load_model(model_path, backbone_name='resnet50', convert=True, nms_threshold=config["nms_threshold"])
        return model

def predict_image(model, raw_image, score_threshold = 0.1, max_detections= 100):
        #predict
        image        = image_utils.preprocess_image(raw_image)
        image, scale = keras_retinanet_image.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
                image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=label_to_name, score_threshold=score_threshold)
        
        return raw_image
