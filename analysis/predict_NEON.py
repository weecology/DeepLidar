import argparse
import sys
from datetime import datetime
import os
import keras
import tensorflow as tf
import cv2
import numpy as np
import copy

#DeepForest
from keras_retinanet import models

#Add to path
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(".")
from DeepForest.utils import generators
from DeepForest import config, postprocessing

#Parse args
mode_parser     = argparse.ArgumentParser(description='Prediction of a new image')
mode_parser.add_argument('--model', help='path to training model',default=None)
mode_parser.add_argument('--output_dir', default="snapshots/images/")

args=mode_parser.parse_args()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

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

def _get_detections(generator, model, score_threshold=0.05, max_detections=300, save_path=None, experiment=None, postprocess= True):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
        experiment    : Comet ML experiment
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        plot_image = generator.retrieve_window()
        plot_image = copy.deepcopy(plot_image[:,:,::-1])

        #Format name and save
        image_name = generator.image_names[i]        
        row = generator.image_data[image_name]             
        lfname = os.path.splitext(row["tile"])[0] + "_" + str(row["window"]) +"raw_image"              
        
        #Skip if missing a component data source
        if raw_image is False:
            print("Empty image, skipping")
            continue
        
        #predict
        image        = generator.preprocess_image(raw_image)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        
        #name image
        image_name = generator.image_names[i]        
        row = generator.image_data[image_name]             
        fname = os.path.splitext(row["tile"])[0] + "_" + str(row["window"])
        
        #drape boxes if they exist
        if len(image_boxes) > 0:
            #get lidar cloud if a new tile, or if not the same tile as previous image.
            if generator.with_lidar:
                if i == 0:
                    generator.load_lidar_tile()
                elif not generator.image_data[i]["tile"] == generator.image_data[i-1]["tile"]:
                    generator.load_lidar_tile()
            
            #The tile could be the full tile, so let's check just the 400 pixel crop we are interested    
            #Not the best structure, but the on-the-fly generator always has 0 bounds
            if hasattr(generator, 'hf'):
                bounds = generator.hf["utm_coords"][generator.row["window"]]    
            else:
                bounds=[]
            
            if generator.with_lidar:
                #Load tile
                if not generator.row["tile"] == generator.previous_image_path:
                    generator.load_lidar_tile()
                #density = Lidar.check_density(generator.lidar_tile, bounds=bounds)
                                
                if postprocess:
                    #find window utm coordinates
                    #print("Bounds for image {}, window {}, are {}".format(generator.row["tile"], generator.row["window"], bounds))
                    pc = postprocessing.drape_boxes(boxes=image_boxes, pc = generator.lidar_tile, bounds=bounds)     
                    
                    #Get new bounding boxes
                    image_boxes = postprocessing.cloud_to_box(pc, bounds)    
                    image_scores = image_scores[:image_boxes.shape[0]]
                    image_labels = image_labels[:image_boxes.shape[0]] 
                    if len(image_boxes)>0:
                        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
                    else:
                        pass
                    #TODO what to do if there are no rows?
                else:
                    pass
                    #print("Point density of {:.2f} is too low, skipping image {}".format(density, generator.row["tile"]))        

        if save_path is not None:
            #draw_annotations(plot_rgb, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(plot_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name,score_threshold=score_threshold)
        
            #name image
            image_name=generator.image_names[i]        
            row=generator.image_data[image_name]             
            fname=os.path.splitext(row["tile"])[0] + "_" + str(row["window"])
        
            #Write RGB
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(fname)), plot_image)
            
            if experiment:
                experiment.log_image(os.path.join(save_path, '{}.png'.format(fname)),name=fname)      

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

#load config
DeepForest_config = config.load_config("../")

trained_models = {"SJER":"/orange/ewhite/b.weinstein/retinanet/20190715_133239/resnet50_30.h5",
                  "TEAK":"/orange/ewhite/b.weinstein/retinanet/20190713_230957/resnet50_40.h5",
                      "NIWO":"/orange/ewhite/b.weinstein/retinanet/20190712_055958/resnet50_40.h5",
                      "MLBS":"/orange/ewhite/b.weinstein/retinanet/20190712_035528/resnet50_40.h5",
                      "All":"/orange/ewhite/b.weinstein/retinanet/20190715_123358/resnet50_40.h5"}

for trained_model in trained_models:
    # load retinanet model
    model_path = trained_models[trained_model]
    model = models.load_model(model_path, backbone_name='resnet50', convert=True, nms_threshold=DeepForest_config["nms_threshold"])
    
    #Make a new dir to hold images
    dirname = datetime.now().strftime("%Y%m%d_%H%M%S")+ trained_model
    save_image_path=os.path.join("..","snapshots", dirname)
    os.mkdir(save_image_path)        
    
    NEON_generator = generators.create_NEON_generator(DeepForest_config["batch_size"], DeepForest_config, name="evaluation")
    all_detections = _get_detections(NEON_generator, model, score_threshold=DeepForest_config["score_threshold"], max_detections=300, save_path=save_image_path, experiment=None)