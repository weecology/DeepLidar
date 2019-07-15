#Test generators
#Test loading in training data
import os
import sys
import glob

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)

#Load modules
from DeepForest import config
from DeepForest.preprocess import NEON_annotations, load_csvs, split_training
from DeepForest.utils import generators

DeepForest_config = config.load_config(dir="..")

def test_load_retraining_data(DeepForest_config):
        data = generators.load_retraining_data(DeepForest_config)
        train, test = split_training(data, DeepForest_config, experiment=None)                
        print("Train shape {}".format(train.shape))
        
def test_load_retraining_data_ablation(DeepForest_config):
        DeepForest_config["training_proportion"]=0.5
        data = generators.load_retraining_data(DeepForest_config)
        train, test = split_training(data, DeepForest_config, experiment=None)        
        print("Train shape {}".format(train.shape))

test_load_retraining_data(DeepForest_config)
test_load_retraining_data_ablation(DeepForest_config)

test_csv_list  =["/Users/Ben/Downloads/fourchannel/TEAK/hand_annotations/2018_TEAK_3_315000_4094000_image_crop.csv"]
def test_load_csvs(test_csv_list):
        df = load_csvs(csv_list = test_csv_list)
        assert df.shape[0] > 0, "data is empty"
        
#test_load_csvs(test_csv_list)

