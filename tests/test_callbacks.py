#test callback
from comet_ml import Experiment
import pytest
import os
import sys
from memory_profiler import profile
precision = 10

import matplotlib.pyplot as plt
import h5py
import pandas as pd

experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=True)
import keras
import tensorflow as tf

fp = open('callbacks.log', 'w+')

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)

from DeepForest import evalmAP, config
from DeepForest.utils.generators import create_NEON_generator
from keras_retinanet  import models
DeepForest_config = config.load_config(dir="..")

site = "TEAK"

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

print('Loading model, this may take a second...')

model = models.load_model("../snapshots/resnet50_28.h5", backbone_name="resnet50", convert=True, nms_threshold=DeepForest_config["nms_threshold"])

@profile(precision=precision, stream=fp)
def test_callback(model, experiment):
    #create the NEON generator 
    NEON_generator = create_NEON_generator(DeepForest_config["batch_size"], DeepForest_config)
    
    average_precisions = evalmAP.evaluate(
        NEON_generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.15,
        max_detections=300,
        save_path="../snapshots/",
        experiment=experiment
    )
    
test_callback(model, experiment)