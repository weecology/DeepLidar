"""
Train a series of models based on a set of sites. 
For example, retrain on each epoch of pretraining data
"""
from comet_ml import Experiment
import sys
import os
from datetime import datetime
import glob
import pandas as pd 
import copy

#insert path 
from DeepForest.config import load_config
from DeepForest.utils.generators import load_retraining_data
from train import main as training_main
from eval import main as eval_main

#load config - clean
original_DeepForest_config = load_config()       

#The following models have been pretrained on all other sites except for the name in the site key
pretraining_models = {
    "SJER":"/orange/ewhite/b.weinstein/retinanet/20190625_133944/resnet50_05.h5",
    "TEAK": "/orange/ewhite/b.weinstein/retinanet/20190625_134530/resnet50_05.h5",
    "NIWO":"/orange/ewhite/b.weinstein/retinanet/20190625_152405/resnet50_05.h5",
    "MLBS": "/orange/ewhite/b.weinstein/retinanet/20190625_133900/resnet50_05.h5"
}

#For each site, match the hand annotations with the pretraining model
results = []
for pretraining_site in pretraining_models:
    
    pretrain_model_path = pretraining_models[pretraining_site]
    print("Running pretraining for  {}".format(pretraining_site))
    
    #load config - clean
    DeepForest_config = copy.deepcopy(original_DeepForest_config)      
    
    ##Replace config file and experiment
    DeepForest_config["hand_annotation_site"] = [pretraining_site]
    DeepForest_config["evaluation_site"] = [pretraining_site]
    
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    experiment.log_parameter("mode","training_grid")   
    experiment.log_parameters(DeepForest_config)    
    DeepForest_config["evaluation_images"] =0         
    
    ###Log experiments
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")        
    experiment.log_parameter("Start Time", dirname)    
    
    ##Make a new dir and reformat args
    save_snapshot_path = DeepForest_config["save_snapshot_path"]+ dirname            
    save_image_path = DeepForest_config["save_image_path"]+ dirname
    os.mkdir(save_snapshot_path)        
    
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)        
    
    #Load retraining data
    data = load_retraining_data(DeepForest_config)
    for x in site:
        DeepForest_config[x]["h5"] = os.path.join(DeepForest_config[x]["h5"],"hand_annotations")
        print(DeepForest_config[x]["h5"])
    
    args = [
        "--epochs", str(DeepForest_config['epochs']),
        "--batch-size", str(DeepForest_config['batch_size']),
        "--backbone", str(DeepForest_config["backbone"]),
        "--score-threshold", str(DeepForest_config["score_threshold"]),
        "--save-path", save_image_path,
        "--snapshot-path", save_snapshot_path,
        "--weights", str(pretrain_model_path)
    ]

    #Run training, and pass comet experiment class
    model = training_main(args=args, data=data, DeepForest_config=DeepForest_config, experiment=experiment)  
    
    #Run eval
    #Always use all hand annotations
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar', log_code=False)
    experiment.log_parameter("mode","evaluation_grid")
    experiment.log_parameters(DeepForest_config)            
        
    args = [
        "--batch-size", str(DeepForest_config['batch_size']),
        '--score-threshold', str(DeepForest_config['score_threshold']),
        '--suppression-threshold', '0.1', 
        '--save-path', 'snapshots/images/', 
        '--model', model, 
        '--convert-model'
    ]
               
    mAP = eval_main(data = data, DeepForest_config = DeepForest_config, experiment = experiment, args = args)
    results.append({"Evaluation Site" : site, "mAP": mAP})
    
results = pd.DataFrame(results)

results.to_csv("analysis/site_grid" + ".csv")        
        