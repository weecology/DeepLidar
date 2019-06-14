#!/usr/bin/env python

import argparse
import os
import sys
from datetime import datetime

import warnings
warnings.simplefilter("ignore")

import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version

#Custom Generators and callbacks
#Path hack
sys.path.insert(0, os.path.abspath('..'))
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.evaluation import neonRecall
from DeepForest.evalmAP import evaluate
from DeepForest import preprocess
from DeepForest.utils.generators import create_NEON_generator

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def parse_args(args):
    """ Parse the arguments.
    """
    
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--model',             help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--suppression-threshold',  help='Permitted overlap among predictions', default=0.2, type=float)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=400)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)

def main(DeepForest_config, args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    #Add seperate dir
    #save time for logging
    dirname=datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path + dirname):
        os.makedirs(args.save_path + dirname)
        
    #Evaluation metrics
    site=DeepForest_config["evaluation_site"]
    
    #create the NEON mAP generator 
    NEON_generator = create_NEON_generator(args.batch_size, DeepForest_config)
    
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model, nms_threshold=DeepForest_config["nms_threshold"])

    #print(model.summary())
        
    #NEON plot mAP
    average_precisions = evaluate(
        NEON_generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=None
    )
    
    return average_precisions
    
if __name__ == '__main__':
    
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    
    #Model list
    trained_models = {"SJER":"/orange/ewhite/b.weinstein/retinanet/20190605_010354/resnet50_30.h5",
                          "TEAK":"/orange/ewhite/b.weinstein/retinanet/20190605_085411/resnet50_30.h5",
                          "NIWO":"/orange/ewhite/b.weinstein/retinanet/20190606_120905/resnet50_50.h5",
                          "All":"/orange/ewhite/b.weinstein/retinanet/20190614_020934/resnet50_30.h5"}
    import pandas as pd
    import numpy as np
    from DeepForest.config import load_config

    DeepForest_config = load_config("..")

    results = []
    for training_site in trained_models:
        #Sites are passed as list object.
        sites = [["TEAK"],["SJER"],["NIWO"]]
        for eval_site in sites:
            
            model  = trained_models[training_site]
            #pass an args object instead of using command line        
            args = [
                "--batch-size", str(DeepForest_config['batch_size']),
                '--score-threshold', str(DeepForest_config['score_threshold']),
                '--suppression-threshold','0.15', 
                '--save-path', '../snapshots/images/', 
                '--model', model, 
                '--convert-model'
            ]
               
            #Run eval
            DeepForest_config["evaluation_site"] = eval_site
            average_precisions = main(DeepForest_config, args)
            
            # print evaluation
            ## print evaluation
            present_classes = 0
            precision = 0
            for label, (average_precision, num_annotations) in average_precisions.items():
                if num_annotations > 0:
                    present_classes += 1
                    precision       += average_precision
            NEON_map = round(precision / present_classes,3)
            
            results.append({"Training Site":training_site, "Evaluation Site": eval_site, "mAP": NEON_map}) 
            
    results = pd.DataFrame(results)
    #model name
    model_name = os.path.splitext(os.path.basename(model))[0]
    
    results.to_csv("cross_site.csv")