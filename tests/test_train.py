#Test loading in training data
import os
import sys

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)

#Load modules
from DeepForest.utils.generators import create_NEON_generator, load_training_data, load_retraining_data, create_h5_generators
from DeepForest import config

DeepForest_config = config.load_config(dir="..")

def test_train():
    data = load_training_data(DeepForest_config)    
    print("Data shape is {}".format(data.shape))
    assert data.shape[0] > 0, "Data is empty"

def test_retrain():
    data = load_retraining_data(DeepForest_config)
    print("Data shape is {}".format(data.shape))    
    assert data.shape[0] > 0, "Data is empty"

test_train()
test_retrain()    