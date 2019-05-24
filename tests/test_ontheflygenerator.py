import pytest
import os
import sys
from memory_profiler import profile
import matplotlib.pyplot as plt

precision = 8

fp = open('onthefly_memory.log', 'w+')

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)
print(parent_path)
from DeepForest import onthefly_generator, preprocess, config

DeepForest_config = config.load_config(dir="..")

site = "TEAK"
tile_xml = "../data/TEAK/annotations/TEAK_044.xml"
base_dir = DeepForest_config[site]["evaluation"]["RGB"]

#Load xml annotations
#data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
#data["site"] = site
#tilename = os.path.splitext(os.path.basename(tile_xml))[0] 

##Create windows
#windows = preprocess.create_windows(data, DeepForest_config, base_dir) 

def test_OnTheFlyGenerator_small(data,windows, DeepForest_config):
    #Create generate
    generator = onthefly_generator.OnTheFlyGenerator(data, windows, DeepForest_config, name="evaluation")
    
    assert generator.size() == 1, "Generate does not have the correct number of images"
    for i in range(1):
        four_channel_image = generator.load_image(i)
        
        print("Image shape of index {} is {}".format(i,four_channel_image.shape))
        fig = plt.figure()
        rgb_image = four_channel_image[:,:,:3]
        plt.imshow(rgb_image[:,:,::-1], origin="upper")
        plt.matshow(four_channel_image[:,:,3], fignum=False,alpha=0.2)
        plt.show()

#test_OnTheFlyGenerator_small(data, windows, DeepForest_config)

base_dir = DeepForest_config[site]["hand_annotations"]["RGB"]
tile_xml = "../data/TEAK/annotations/2018_TEAK_3_315000_4094000_image_crop.xml"
#Load xml annotations
data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
data["site"] = site
tilename = os.path.splitext(os.path.basename(tile_xml))[0] 

#Create windows
windows = preprocess.create_windows(data, DeepForest_config, base_dir) 

@profile(precision=precision, stream=fp)
def test_OnTheFlyGenerator_large(data, windows, DeepForest_config):
    #Create generate
    generator = onthefly_generator.OnTheFlyGenerator(data, windows, DeepForest_config)
    
    assert generator.size() == 256, "Generate does not have the correct number of images"
    for i in range(256):
        image = generator.load_image(i)
        print("Image shape of index {} is {}".format(i,image.shape))

test_OnTheFlyGenerator_large(data, windows, DeepForest_config)