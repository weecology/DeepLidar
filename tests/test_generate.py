import pytest
import os
import sys
from memory_profiler import profile

precision = 10

fp = open('memory_profiler_basic_mean.log', 'w+')

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)
from DeepForest import Generate, preprocess, config
DeepForest_config = config.load_config(dir="..")

site = "TEAK"
tile_xml = "../data/TEAK/annotations/2018_TEAK_3_315000_4094000_image_crop.xml"

@profile(precision=precision, stream=fp)
def test_Generate_xml(tile_xml, DeepForest_config, site):
    #Create generate
    csv_file, h5_file = Generate.run(tile_xml, DeepForest_config, site)
    
    #Check csv has correct length
    #assert
    
    #Check h5 has correct length
    #assert 

#test_Generate_xml(data, DeepForest_config,site)

tile_csv = "/Users/ben/Documents/DeepLidar/data/TEAK/training/NEON_D17_TEAK_DP1_324000_4104000_classified_point_cloud_colorized.csv"
def test_Generate_csv(tile_csv, DeepForest_config, site):
    csv_file, h5_file  = Generate.run(tile_csv, DeepForest_config=DeepForest_config, site=site)

test_Generate_csv(tile_csv, DeepForest_config, site)