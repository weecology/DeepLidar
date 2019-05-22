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
print(parent_path)
from DeepForest import onthefly_generator, preprocess, config

DeepForest_config = config.load_config(dir="..")

site = "TEAK"
tile_xml = "../data/TEAK/annotations/2018_TEAK_3_315000_4094000_image_crop.xml"
base_dir = DeepForest_config[site]["hand_annotations"]["RGB"]

#Load xml annotations
data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
data["site"] = site
tilename = os.path.splitext(os.path.basename(tile_xml))[0] 

#Create windows
windows = preprocess.create_windows(data, DeepForest_config, base_dir) 

@profile(precision=precision, stream=fp)
def test_OnTheFlyGenerator(data,windows, DeepForest_config):
    #Create generate
    generator = onthefly_generator.OnTheFlyGenerator(data, windows, DeepForest_config)
    
    assert generator.size() == 256, "Generate does not have the correct number of images"
    for i in range(20):
        image = generator.load_image(i)
        print("Image shape of index {} is {}".format(i,image.shape))

test_OnTheFlyGenerator(data, windows, DeepForest_config)