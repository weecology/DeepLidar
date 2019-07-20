'''
Preprocess data
Loading, Standardizing and Filtering the raw data to dimish false positive labels 
'''

import pandas as pd
import glob
import os
import random
import xmltodict
import numpy as np
import rasterio
from PIL import Image
import slidingwindow as sw
import itertools
import re
import warnings

def load_csvs(h5_dir=None, csv_list=None):
    """
    Read preprocessed csv files generated by Generate.py run method
    h5_dir: Directory to search for csv files
    csv_list: Optional list of csvs to directly specify csvs
    """
    if h5_dir:
        #If a single file, read, if a a dir, loop through files
        if os.path.isdir(h5_dir):
            #Gather list of csvs
            data_paths = glob.glob(h5_dir + "/*.csv")
            dataframes = (pd.read_csv(f, index_col=None) for f in data_paths)
            data = pd.concat(dataframes, ignore_index=True)      
    else:
        dataframes = (pd.read_csv(f, index_col=None) for f in csv_list)
        data = pd.concat(dataframes, ignore_index=True)      
 
    return data
    
    
def load_data(data_dir, res, lidar_path):
    '''
    Load csv training data from TreeSegmentation R package
    data_dir: path to .csv files. Optionall can be a path to a specific .csv file.
    res: Cell resolution of the rgb imagery
    '''
    
    if(os.path.splitext(data_dir)[-1] == ".csv"):
        data=pd.read_csv(data_dir, index_col=0)
    else:
        #Gather list of csvs
        data_paths = glob.glob(data_dir+"/*.csv")
        dataframes = (pd.read_csv(f, index_col=0) for f in data_paths)
        data = pd.concat(dataframes, ignore_index=False)
    
    #Modify indices, which came from R, zero indexed in python
    data=data.set_index(data.index.values-1)
    data.numeric_label=data.numeric_label-1
    
    #Remove xmin==xmax
    data=data[data.xmin!=data.xmax]    
    data=data[data.ymin!=data.ymax]    

    ##Create bounding coordinates with respect to the crop for each box
    #Rescaled to resolution of the cells.Also note that python and R have inverse coordinate Y axis, flipped rotation.
    data['origin_xmin']=(data['xmin']-data['tile_xmin'])/res
    data['origin_xmax']=(data['xmin']-data['tile_xmin']+ data['xmax']-data['xmin'])/res
    data['origin_ymin']=(data['tile_ymax']-data['ymax'])/res
    data['origin_ymax']= (data['tile_ymax']-data['ymax']+ data['ymax'] - data['ymin'])/res  
        
    #Check for lidar tiles
    data=check_for_lidar(data=data, lidar_path=lidar_path)
    
    #Check for remaining data
    assert(data.shape[0] > 0),"No training data remaining after ingestion, check lidar paths"
    
    return(data)

def load_xmls(path, h5_dirname, rgb_tile_dir, rgb_res):
    """
    Extend load_xml to a possible directory
    path: input directory or path of the .xml annotations
    dirname: where to search the h5 dir
    """
    #Load xml annotations and find the directory of .tif files
    if os.path.isdir(path):
        xmls = glob.glob(os.path.join(path,"*.tif"))
        
        #set xml dir, assume its in annotation folder
        annotation_dir = os.path.join(os.path.dirname(os.path.dirname(path)),"annotations")
        annotation_xmls = [os.path.splitext(os.path.basename(x))[0] + ".xml" for x in xmls]
        full_xml_path = [os.path.join(annotation_dir, x ) for x in annotation_xmls]
        
        xml_data = []
        for x in full_xml_path:
            xml_data.append(load_xml(x, dirname=rgb_tile_dir, res=rgb_res))            
        data = pd.concat(xml_data)
    else:
        
        data = load_xml(path, dirname=rgb_tile_dir, res=rgb_res)
    
        return data
    
def load_xml(path, dirname, res):
    """
    Load a single .xml annotations
    """
    #parse
    with open(path) as fd:
        doc = xmltodict.parse(fd.read())
    
    #grab xml objects
    try:
        tile_xml=doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(e,path, doc["annotation"]))
        
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    label=[]
    
    if type(tile_xml) == list:
        treeID=np.arange(len(tile_xml))
        
        #Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        #One tree
        treeID=0
        
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])        
        
    rgb_path = doc["annotation"]["filename"]
    
    #bounds
    #read in tile to get dimensions
    full_path=os.path.join(dirname, rgb_path)

    with rasterio.open(full_path) as dataset:
        bounds=dataset.bounds         
        
    frame=pd.DataFrame({"treeID":treeID,
                        "xmin":xmin,"xmax":xmax,
                        "ymin":ymin,"ymax":ymax,
                        "rgb_path":rgb_path,
                        "label":label,
                        "numeric_label":0,
                        "tile_xmin":bounds.left,
                        "tile_xmax":bounds.right,
                        "tile_ymin":bounds.bottom,
                        "tile_ymax":bounds.top}
                       )

    #Modify indices, which came from R, zero indexed in python
    frame=frame.set_index(frame.index.values)

    ##Match expectations of naming, no computation needed for hand annotations
    frame['origin_xmin'] = frame["xmin"].astype(float)
    frame['origin_xmax'] = frame["xmax"].astype(float)
    frame['origin_ymin'] = frame["ymin"].astype(float)
    frame['origin_ymax'] = frame["ymax"].astype(float)
    
    return(frame)

def compute_windows(image, pixels=250, overlap=0.05):
    try:
        im = Image.open(image)
    except:
        return None
    numpy_image = np.array(im)    
    windows = sw.generate(numpy_image, sw.DimOrder.HeightWidthChannel, pixels,overlap)
    
    return(windows)

def retrieve_window(numpy_image,index,windows):
    crop=numpy_image[windows[index].indices()]
    return(crop)

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def check_for_lidar(data,lidar_path):
    lidar_tiles = data.lidar_path.unique()
    
    lidar_exists = []
    for x in list(lidar_tiles):
        does_exist = os.path.exists(os.path.join(lidar_path,x))
        lidar_exists.append(does_exist)
    
    #Filter data based on matching lidar tiles
    matching_lidar = list(lidar_tiles[lidar_exists])
    data = data[data.lidar_path.isin(matching_lidar)]
    
    return data

def split_training(data, DeepForest_config, experiment):
    
    '''
    Divide windows into training and testing split
    data: a pandas dataframe created by DeepForest Generate Run
    experiment: The comet experiment object
    '''
    
    #reduce the data frame into tiles and windows
    windowdf = data[["tile","window","site"]]
    data = windowdf.drop_duplicates()
    
    #More than one tile in training data?
    single_tile =  len(data.tile.unique()) == 1
    
    if DeepForest_config["evaluation_images"] == 0:
        training= data
        evaluation = None
    else:
        if single_tile: 
                #Select n% as validation
                msk = np.random.rand(len(data)) < 1-(float(DeepForest_config["validation_percent"])/100)
                training = data[msk]
                evaluation = data[~msk]  
        else:
                eval_tile = data.tile.unique()[1]
                evaluation = data[data["tile"] == eval_tile]
                training = data[~(data["tile"] == eval_tile)]
    
    #If selecting training samples as a proportion
    if not DeepForest_config["training_proportion"]==1:
            DeepForest_config["training_images"] = int(training.shape[0] * DeepForest_config["training_proportion"])
            print("Superceding number of training images {} based on training proportion {}".format(DeepForest_config["training_images"],DeepForest_config["training_proportion"]))
            if experiment:
                experiment.log_parameter("final_training_images",DeepForest_config["training_images"])
    
    #Select n Training samples
    if not DeepForest_config["training_images"]=="All":
        num_training_images = DeepForest_config["training_images"]
        
        if num_training_images > training.shape[0]:
            raise ValueError("Number of training samples greater than available windows")
            
        #Optional shuffle
        if DeepForest_config["shuffle_training"]:
            training = training.sample(frac=1)
            
        #Select subset of training windows
        training = training.iloc[0:num_training_images]
        
        #Ensure training is sorted by image
        training.sort_values(by="tile", inplace=True)    
        
        #Shuffle tile order is needed
        groups = [df for _, df in training.groupby('tile')]
        random.shuffle(groups)    
        training = pd.concat(groups,sort=False).reset_index(drop=True)
        
    else:
        if DeepForest_config["shuffle_training"]:
            #Shuffle tile order is needed
            groups = [df for _, df in training.groupby('tile')]
            random.shuffle(groups)    
            training = pd.concat(groups,sort=False).reset_index(drop=True)
        
    #evaluation samples
    if not DeepForest_config["evaluation_images"] in ["All",0]:
        num_evaluation_images = DeepForest_config["evaluation_images"]
    
        #Optional shuffle
        if DeepForest_config["shuffle_evaluation"]:
            evaluation = evaluation.sample(frac=1)
            
        #Select subset of evaluation windows
        evaluation=evaluation.iloc[0:num_evaluation_images]
    
    #Make sure we have training data, if not raise error.
    assert training.shape[0] > 0, "Training data is empty"
    
    return([training, evaluation])
    
def NEON_annotations(DeepForest_config):
    '''
    Create a keras generator for the hand annotated tower plots. Used for the mAP and recall callback.
    '''
    annotations=[]    
    for site in DeepForest_config["evaluation_site"]:
        
        #Set directory to the plots 
        plot_dir = DeepForest_config[site]["evaluation"]["RGB"]
        
        #up one level is assumed to be annotation dir
        base_dir = os.path.dirname(os.path.dirname(plot_dir))
        annotation_dir = os.path.join(base_dir ,"annotations")
        
        #Find all images and annotations for each with full path
        images_to_find_xml = glob.glob(os.path.join(plot_dir, "*.tif"))
        corresponding_xml=[os.path.splitext(os.path.basename(x))[0] + ".xml"  for x in images_to_find_xml]
        full_path_xml = []
        
        for x in corresponding_xml:
            full_path_xml.append(os.path.join(annotation_dir,x))
            
        glob_path = os.path.join(annotation_dir, "*.xml")
        available_xmls = glob.glob(glob_path)
        
        #matched xmls
        matched_xmls = set(full_path_xml) & set(available_xmls)
        
        for xml in matched_xmls:
            xml_data = load_xml(xml, dirname=plot_dir, res=DeepForest_config["rgb_res"])
            #add site
            xml_data["site"] = site
            annotations.append(xml_data)

    data = pd.concat(annotations)
    
    #Compute list of sliding windows, assumed that all objects are the same extent and resolution
    #TO DO this could be an area of concern.
    image_path = os.path.join(plot_dir, data[data["site"]==site].rgb_path.unique()[0])
    windows = compute_windows(image=image_path, pixels=DeepForest_config["patch_size"], overlap=DeepForest_config["patch_overlap"])
    
    #Create dictionary of windows for each image
    tile_windows = {}
    all_images = list(data.rgb_path.unique())
    tile_windows["tile"] = all_images
    tile_windows["window"] = np.arange(0, len(windows))
    
    #Expand all combinations
    tile_data = expand_grid(tile_windows)    
    
    #merge site data
    merge_site = data[["rgb_path","site"]].drop_duplicates()
    merge_site.columns = ["tile","site"]
    tile_data = tile_data.merge(merge_site)
    
    return [data, tile_data]

def create_windows(data, DeepForest_config, base_dir):
    """
    Generate windows for a specific tile
    base_dir: Location of the RGB image data
    """
    #Compute list of sliding windows, assumed that all objects are the same extent and resolution     
    sample_tile = data.rgb_path.iloc[0]
    image_path=os.path.join(base_dir, sample_tile)
    windows=compute_windows(image=image_path, pixels=DeepForest_config["patch_size"], overlap=DeepForest_config["patch_overlap"])
    
    #if none
    if windows is None:
        return None
    
    #Compute Windows
    #Create dictionary of windows for each image
    tile_windows={}
    
    all_images=list(data.rgb_path.unique())

    tile_windows["tile"] = all_images
    tile_windows["window"]=np.arange(0, len(windows))
    
    #Expand grid
    tile_data = expand_grid(tile_windows)    
    
    #Merge with the site variable
    merge_site = data[["rgb_path","site"]].drop_duplicates()
    merge_site.columns = ["tile","site"]
    tile_data = tile_data.merge(merge_site)
    
    return(tile_data)