"""
Train a series of models based on a set of sites. 
For example, retrain on each epoch of pretraining data
"""
from comet_ml import Experiment
import keras
import tensorflow as tf
import sys
import os
from datetime import datetime
import glob
import pandas as pd 
import copy
import numpy as np

from keras_retinanet import models
from keras_retinanet .models.retinanet import retinanet_bbox

#insert path 
sys.path.insert(0, os.path.abspath('..'))
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data, create_h5_generators
from train import main as training_main
from prcurve import main as eval_main

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#load config - clean
original_DeepForest_config = load_config("..")       

#The following models have been pretrained on all other sites except for the name in the site key
pretraining_models = {
    "SJER": "/orange/ewhite/b.weinstein/retinanet/20190711_121755/resnet50_05.h5",
    "TEAK": "/orange/ewhite/b.weinstein/retinanet/20190713_102002/resnet50_04.h5",
    "NIWO":"/orange/ewhite/b.weinstein/retinanet/20190711_195718/resnet50_05.h5",
    "MLBS":  "/orange/ewhite/b.weinstein/retinanet/20190711_121214/resnet50_05.h5"
}

#For each site, match the hand annotations with the pretraining model
results = []
for pretraining_site in pretraining_models:
    
    #For each site run a portion of the training data
    for x in np.arange(4):
        for proportion_data in [0, 0.01, 0.05,0.25,0.5,0.75,1]:
            pretrain_model_path = pretraining_models[pretraining_site]
            print("Running pretraining for  {}".format(pretraining_site))
            
            #load config - clean
            DeepForest_config = copy.deepcopy(original_DeepForest_config)      
            
            ##Replace config file and experiment
            DeepForest_config["hand_annotation_site"] = [pretraining_site]
            DeepForest_config["evaluation_site"] = [pretraining_site]
            DeepForest_config["batch_size"] = 40
            DeepForest_config["epochs"] = 30
            DeepForest_config["save_image_path"] =  None
            
            experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
            experiment.log_parameter("mode","ablation")   
            DeepForest_config["evaluation_images"] =0         
            
            #set training images, as a function of the number of training windows
            DeepForest_config["training_proportion"] = proportion_data    
            
            ###Log experiments
            dirname = datetime.now().strftime("%Y%m%d_%H%M%S")        
            experiment.log_parameter("Start Time", dirname)    
            experiment.log_parameters(DeepForest_config)    
            
            #Load retraining data
            data = load_retraining_data(DeepForest_config)
            for x in [pretraining_site]:
                DeepForest_config[x]["h5"] = os.path.join(DeepForest_config[x]["h5"],"hand_annotations")
                print(DeepForest_config[x]["h5"])
            
            if not proportion_data == 0:
                #Run training, and pass comet experiment class
                #start training
                train_generator, validation_generator = create_h5_generators(data, DeepForest_config=DeepForest_config)
                
                print('Loading model, this may take a secondkeras-retinanet.\n')
                model            = models.load_model(pretrain_model_path, backbone_name=DeepForest_config["backbone"])
                training_model   = model
                prediction_model = retinanet_bbox(model=model, nms_threshold=DeepForest_config["nms_threshold"])
                
                history = training_model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
                    epochs=DeepForest_config["epochs"],
                    verbose=2,
                    shuffle=False,
                    workers=DeepForest_config["workers"],
                    use_multiprocessing=DeepForest_config["use_multiprocessing"],
                    max_queue_size=DeepForest_config["max_queue_size"])
                
                model_path = training_main(args=args, data=data, DeepForest_config=DeepForest_config, experiment=experiment)  
                num_trees = experiment.get_parameter("Number of Training Trees")
                
            else: 
                model_path = pretrain_model_path
                num_trees = 0
                
            #Run eval
            #Always use all hand annotations
            experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
            experiment.log_parameter("mode","ablation_evaluation")
            experiment.log_parameters(DeepForest_config)            
                
            args = [
                "--batch-size", str(DeepForest_config['batch_size']),
                '--score-threshold', str(DeepForest_config['score_threshold']),
                '--suppression-threshold', '0.1', 
                '--save-path', 'snapshots/images/', 
                '--model', model_path, 
                '--convert-model'
            ]
                 
            # load the model just once
            keras.backend.tensorflow_backend.set_session(get_session())
            print('Loading model, this may take a second...')
            model = models.load_model(model_path, backbone_name="resnet50", convert=True, nms_threshold=DeepForest_config["nms_threshold"])
    
            recall, precision  = eval_main(DeepForest_config = DeepForest_config, args = args, model=model)
            results.append({"Number of Trees": num_trees, "Proportion":proportion_data,"Evaluation Site" : pretraining_site, "Recall": recall,"Precision": precision})
            
results = pd.DataFrame(results)

results.to_csv("ablation_{}".format(dirname) + ".csv")        
