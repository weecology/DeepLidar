#test multi-gpu settings
#test callback
from comet_ml import Experiment
import pytest
import os
import sys
from memory_profiler import profile
precision = 10

#Path hack
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)
from train import create_models 
from DeepForest.utils import generators
from DeepForest import config

#retinanet
from keras_retinanet import models
import cProfile
fp = open('multigpu.log', 'w+')

@profile(precision=precision, stream=fp)
def test_multigpu_training():
    
    experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='deeplidar',log_code=True)
    
    DeepForest_config = config.load_config(dir="..")    
    DeepForest_config["save_image_path"] = "../snapshots/"
    
    data = generators.load_retraining_data(DeepForest_config)   
    train_generator, validation_generator = generators.create_h5_generators(data, DeepForest_config=DeepForest_config)
    
    #imagenet pretraining weights
    backbone = models.backbone(DeepForest_config["backbone"])
    weights = backbone.download_imagenet()
    
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=train_generator.num_classes(),
        weights=weights,
        multi_gpu=DeepForest_config["num_GPUs"],
        freeze_backbone=False,
        nms_threshold=DeepForest_config["nms_threshold"],
        input_channels=DeepForest_config["input_channels"]
    )
    
    #start training
    history = training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
        epochs=DeepForest_config["epochs"],
        verbose=2,
        shuffle=False,
        workers=DeepForest_config["workers"],
        use_multiprocessing=DeepForest_config["use_multiprocessing"],
        max_queue_size=DeepForest_config["max_queue_size"],
        experiment = experiment
    )

test_multigpu_training()
cProfile.run('test_multigpu_training', 'test_multi_training.prof')
