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
from DeepForest.onthefly_generator import OnTheFlyGenerator
from DeepForest.preprocess import NEON_annotations, load_csvs, split_training
from DeepForest import Generate
from DeepForest.h5_generator import H5Generator
from DeepForest.utils import image_utils
from DeepForest.h5_generator import H5Generator

DeepForest_config = config.load_config(dir="..")

test_csv_list  =["/Users/Ben/Downloads/fourchannel/TEAK/hand_annotations/2018_TEAK_3_315000_4094000_image_crop.csv"]
def test_load_csvs(test_csv_list):
        df = load_csvs(csv_list = test_csv_list)
        assert df.shape[0] > 0, "data is empty"
        
test_load_csvs(test_csv_list)

