#Test preprocessing.py
import pytest
import os
import sys

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)

from DeepForest import preprocess, config
from DeepForest.utils import generators
DeepForest_config = config.load_config(dir="..")

site = "MLBS"
tile_xml = "../data/MLBS/annotations/MLBS_064.xml"
base_dir = DeepForest_config[site]["evaluation"]["RGB"]

#Load xml annotations
def test_load_xml():
    data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
    assert data.shape[0] > 0, "Data is empty"

#test_load_xml()

DeepForest_config["evaluation_site"]=["NIWO"]
DeepForest_config["training_proportion"]=1

def test_split_training(DeepForest_config):
    data = generators.load_retraining_data(DeepForest_config)   
    train, test = preprocess.split_training(data, DeepForest_config=DeepForest_config, experiment=None)
    
    #Has data
    assert train.shape[0] > 0, "Train data is empty"
    print(train.shape)
    
#Test full proportion
test_split_training(DeepForest_config)

#Test tiny proportion
DeepForest_config["shuffle_training"]=False
DeepForest_config["training_proportion"]=0.01
test_split_training(DeepForest_config)

#Shuffle training
DeepForest_config["shuffle_training"]=True
test_split_training(DeepForest_config)


