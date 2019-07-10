#!/usr/bin/env python


import argparse
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(".")
import warnings
warnings.simplefilter("ignore")

import keras
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version

#Custom Generators and callbacks
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.evaluation import neonRecall
from DeepForest.evalmAP import evaluate_pr
from DeepForest import preprocess
from DeepForest.utils.generators import create_NEON_generator
from eval import parse_args, get_session

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
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    recall, precision = evaluate_pr(
        NEON_generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path + dirname,
        experiment=None
    )
    
    return [recall, precision]
    
if __name__ == '__main__':
    
    import argparse
    
    #Set training or training
    mode_parser     = argparse.ArgumentParser(description='Retinanet training or finetuning?')
    mode_parser.add_argument('--saved_model', help='train or retrain?' )    
    mode = mode_parser.parse_args()
    
    import pandas as pd
    import numpy as np
    from DeepForest.config import load_config

    DeepForest_config = load_config("..")
    
    NIWO_list = [
        "/orange/ewhite/b.weinstein/retinanet/20190709_122336/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_130254/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_135128/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_143051/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_150638/resnet50_40.h5"
    ]
        
    TEAK_list = [
        "/orange/ewhite/b.weinstein/retinanet/20190709_161527/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_171539/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_175910/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_191727/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190709_200228/resnet50_40.h5"
    ]

    SJER_list = [
        "/orange/ewhite/b.weinstein/retinanet/20190709_223205/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_100237/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_100252/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_100329/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_100347/resnet50_40.h5"
    ]

    MLBS_list = [
        "/orange/ewhite/b.weinstein/retinanet/20190710_122300/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_122307/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_122310/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_122313/resnet50_40.h5",
        "/orange/ewhite/b.weinstein/retinanet/20190710_122316/resnet50_40.h5"
    ]
    
    trained_models = {
        "NIWO": NIWO_list,
        "SJER": SJER_list,
        "TEAK": TEAK_list,
        "MLBS": MLBS_list}
    
    results = []    
    for training_site in trained_models:
        trained_model_list = trained_models[training_site]
        DeepForest_config["evaluation_site"] = [training_site]
        
        #Loop through trained models per site to create confidence intervals
        for trained_model in trained_model_list:
            for score_threshold in np.arange(0, 1, 0.1):
                #pass an args object instead of using command line        
                args = [
                    "--batch-size", str(DeepForest_config['batch_size']),
                    '--score-threshold', str(score_threshold),
                    '--suppression-threshold','0.15', 
                    '--save-path', 'snapshots/images/', 
                    '--model', trained_model, 
                    '--convert-model'
                ]
                   
                #Run eval
                recall, precision = main(DeepForest_config, args)
                results.append({"Site":training_site,"Model":trained_model, "Threshold": score_threshold, "Recall": recall, "Precision": precision})
            
    results = pd.DataFrame(results)
        
    #model name
    results.to_csv("prcurve_data" + ".csv")