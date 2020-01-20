#plot annotations
import argparse
import sys
from datetime import datetime
import os
import cv2
import numpy as np
import copy

#DeepForest
from keras_retinanet import models
from keras_retinanet.utils.visualization import draw_annotations

#Add to path
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(".")
from DeepForest.utils import generators
from DeepForest import config

def draw_box(image, box, color, thickness=1):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)
    
def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.15):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        draw_box(image, boxes[i, :], color=[0,0,0])
        
def plot_annotations(generator, save_path=None):

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        plot_image = generator.retrieve_window()
        plot_image = copy.deepcopy(plot_image[:,:,::-1])
        
        #Skip if missing a component data source
        if raw_image is False:
            print("Empty image, skipping")
            continue
        
        if save_path is not None:
            draw_annotations(plot_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
        
            #name image
            image_name=generator.image_names[i]        
            row=generator.image_data[image_name]             
            fname=os.path.splitext(row["tile"])[0] + "_" + str(row["window"])
        
            #Write RGB
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(fname)), plot_image)

#load config
DeepForest_config = config.load_config("../")

#Make a new dir to hold images
dirname = datetime.now().strftime("%Y%m%d_%H%M%S")
save_image_path=os.path.join("..","snapshots", dirname)
os.mkdir(save_image_path)            

DeepForest_config["evaluation_site"] = ["MLBS","NIWO","TEAK","SJER"]

NEON_generator = generators.create_NEON_generator(DeepForest_config["batch_size"], DeepForest_config, name="evaluation")
plot_annotations(NEON_generator,save_path=save_image_path)