import pytest
import os
import sys
from memory_profiler import profile
import matplotlib.pyplot as plt

precision = 8

fp = open('h5_memory.log', 'w+')

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)
print(parent_path)
from DeepForest import h5_generator, preprocess, config
from DeepForest.utils import generators, image_utils

DeepForest_config = config.load_config(dir="..")

#Load xml annotations
tiles = generators.load_retraining_data(DeepForest_config)    
train, test  = generators.split_tiles(tiles,DeepForest_config)

@profile(precision=precision, stream=fp)
def test_h5_generator(train, DeepForest_config):
    #Training Generator
    generator = h5_generator.H5generator(train)  
    
    for i in range(len(generator)):
        inputs, targets = generator.__getitem__(i)
        
        assert len(targets)==2, "targets has incorrect length"
        
        assert inputs.shape == (DeepForest_config["batch_size"], DeepForest_config["patch_size"], DeepForest_config["patch_size"], DeepForest_config["input_channels"] )

test_h5_generator(tiles, DeepForest_config)