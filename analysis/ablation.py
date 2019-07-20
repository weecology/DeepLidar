"""
Ablation study to test the effect of the % of training data on precision and recall. 
Run several times and save the output to a pandas frame
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
import gc

from keras_retinanet import models
from keras_retinanet .models.retinanet import retinanet_bbox
from keras_retinanet .callbacks import RedirectModel

#insert path 
sys.path.insert(0, os.path.abspath('..'))
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data, create_h5_generators
from train import create_models
from prcurve import main as eval_main

#load config - clean
DeepForest_config = load_config("..")       

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

def train(pretrain_model_path, data, proportion_data, DeepForest_config):
    ###Log experiments
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    
    #make snapshot dir            
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")  
    experiment.log_parameter("Start Time", dirname)                
    save_snapshot_path=DeepForest_config["save_snapshot_path"] + dirname            
    os.mkdir(save_snapshot_path)   
    
    ##Replace config file and experiment
    DeepForest_config["batch_size"] = 20
    DeepForest_config["epochs"] = 40
    DeepForest_config["save_image_path"] =  None
    experiment.log_parameter("mode","ablation")   
    DeepForest_config["evaluation_images"] = 0         
    
    #set training images, as a function of the number of training windows
    DeepForest_config["training_proportion"] = proportion_data     
    experiment.log_parameters(DeepForest_config)                

    #Create model
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
                    num_classes=1,
                   weights=pretrain_model_path,
                   multi_gpu=DeepForest_config["num_GPUs"],
                   freeze_backbone=False,
                   nms_threshold=DeepForest_config["nms_threshold"],
                   input_channels=DeepForest_config["input_channels"]
    )


    if not proportion_data == 0:
        #Run training, and pass comet experiment class
        #start training
        train_generator, validation_generator = create_h5_generators(data, DeepForest_config=DeepForest_config)     
        
        #ensure directory created first; otherwise h5py will error after epoch.
        history = training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
            epochs=DeepForest_config["epochs"],
            verbose=2,
            shuffle=False,
            workers=DeepForest_config["workers"],
            use_multiprocessing=DeepForest_config["use_multiprocessing"],
            max_queue_size=DeepForest_config["max_queue_size"])
        
        num_trees = train_generator.total_trees
    else: 
        num_trees = 0
    
    #Log trees
    experiment.log_parameter("Number of Training Trees", num_trees)                   
    
    return prediction_model, num_trees

def evaluation(prediction_model, results, DeepForest_config, num_trees):
    print("Evaluation")
    #Run eval
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    experiment.log_parameter("mode","ablation_evaluation")
    experiment.log_parameters(DeepForest_config)            
        
    args = [
        "--batch-size", str(DeepForest_config['batch_size']),
        '--score-threshold', str(DeepForest_config['score_threshold']),
        '--suppression-threshold', '0.1', 
        '--save-path', 'snapshots/images/', 
    ]
    
    recall, precision  = eval_main(DeepForest_config = DeepForest_config, args = args, model=prediction_model)
    results.append({"Number of Trees": num_trees, "Proportion":proportion_data,"Evaluation Site" : pretraining_site, "Recall": recall,"Precision": precision})
    
    return results

if __name__ == "__main__":
    #The following models have been pretrained on all other sites except for the name in the site key
    pretraining_models = {
        "SJER": "/orange/ewhite/b.weinstein/retinanet/20190719_120547/resnet50_05.h5",
        "TEAK": "/orange/ewhite/b.weinstein/retinanet/20190713_102002/resnet50_04.h5"
        #"NIWO":"/orange/ewhite/b.weinstein/retinanet//resnet50_05.h5",
        #"MLBS":  "/orange/ewhite/b.weinstein/retinanet//resnet50_05.h5"
    }
    
    #For each site, match the hand annotations with the pretraining model
    results = []
    for pretraining_site in pretraining_models:  
        
        pretrain_model_path = pretraining_models[pretraining_site]
        print("Running pretraining for {}".format(pretraining_site))
        
        #Load retraining data
        DeepForest_config["hand_annotation_site"] = [pretraining_site]
        DeepForest_config["evaluation_site"] = [pretraining_site]
        DeepForest_config["shuffle_training"] = True
        
        data = load_retraining_data(DeepForest_config)
        for x in [pretraining_site]:
            DeepForest_config[x]["h5"] = os.path.join(DeepForest_config[x]["h5"],"hand_annotations")
            print(DeepForest_config[x]["h5"])
            
        #Load pretraining model
        print('Loading model, this may take a secondkeras-retinanet.\n')
        
        backbone = models.backbone(DeepForest_config["backbone"])         
        
        #For each site run a portion of the training data
        for x in np.arange(2):
            for proportion_data in [0, 0.01, 0.05,0.25,0.5,0.75,1]:
                prediction_model, num_trees = train(pretrain_model_path, data, proportion_data, DeepForest_config)
                results = evaluation(prediction_model, results, DeepForest_config, num_trees)
                del(prediction_model)
                gc.collect()
    
    #Wrap together the results            
    results = pd.DataFrame(results)
    
    #give time stamp in case multiple running
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")      
    results.to_csv("ablation_{}".format(timestamp) + ".csv")        
