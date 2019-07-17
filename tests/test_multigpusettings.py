#test multi-gpu settings
#test callback
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
    
    DeepForest_config = config.load_config(dir="..")    
    data = generators.load_retraining_data(DeepForest_config)   
    train_generator, validation_generator = generators.create_h5_generators(args, data, DeepForest_config=DeepForest_config)
    
    #imagenet pretraining weights
    backbone = models.backbone(args.backbone)
    weights = backbone.download_imagenet()
    
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,
        num_classes=train_generator.num_classes(),
        weights=weights,
        multi_gpu=DeepForest_config["num_gpu"],
        freeze_backbone=False,
        nms_threshold=DeepForest_config["nms_threshold"],
        input_channels=DeepForest_config["input_channels"]
    )
    
    #start training
    history = training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size()/DeepForest_config["batch_size"],
        epochs=args.epochs,
        verbose=2,
        shuffle=False,
        callbacks=callbacks,
        workers=DeepForest_config["workers"],
        use_multiprocessing=DeepForest_config["use_multiprocessing"],
        max_queue_size=DeepForest_config["max_queue_size"])

test_multigpu_training()
cProfile.run('test_multigpu_training', 'test_multi_training.prof')
