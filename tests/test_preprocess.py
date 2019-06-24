#Test preprocessing.py
import pytest
import os
import sys

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)
from DeepForest import preprocess, config
DeepForest_config = config.load_config(dir="..")

site = "MLBS"
tile_xml = "../data/MLBS/annotations/MLBS_064.xml"
base_dir = DeepForest_config[site]["evaluation"]["RGB"]

#Load xml annotations
def test_load_xml():
    data = preprocess.load_xml(path=tile_xml, dirname=base_dir, res=DeepForest_config["rgb_res"])
    assert data.shape[0] > 0, "Data is empty"

test_load_xml()