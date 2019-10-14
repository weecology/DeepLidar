# Geographic Generalization in Airborne RGB Deep Learning Tree Detection

Ben. G. Weinstein, Sergio Marconi, Stephanie Bohlman, Alina Zare, Ethan White

# Summary
DeepLidar is a keras retinanet implementation for predicting individual tree crowns in RGB imagery. 

## How can I train new data?

DeepLidar uses a semi-supervised framework for model training. For generating lidar-derived training data see (). I recommend using a conda environments to manage python dependencies. 

1. Create conda environment and install dependencies

```
conda env create --name DeepForest -f=generic_environment.yml
```

Clone the fork of the retinanet repo and install in local environment

```
conda activate DeepForest
git clone https://github.com/bw4sz/keras-retinanet
cd keras-retinanet
pip install .
```

2. Update config paths

All paths are hard coded into [_config.yml](https://github.com/weecology/DeepLidar/blob/master/_config.yml)

3. Train new model with new hand annotations

```
python train.py --retrain
```

# How can I use pre-built models to predict new images.

Check out a demo ipython notebook: https://github.com/weecology/DeepLidar/tree/master/demo

# Where are the data?

The Neon Trees Benchmark dataset is soon to be published. All are welcome to use it. Currently under curation (in progress): https://github.com/weecology/NeonTreeEvaluation/

For a static version of the dataset that reflects annotations at the time of submission, see dropbox link [here](https://www.dropbox.com/s/yjrhs8b7ocbw6ji/static.zip?dl=0)

## Published articles

Our first article was published in *Remote Sensing* and can be found [here](https://www.mdpi.com/2072-4292/11/11/1309). 

This codebase is constantly evolving and improving. To access the code at the time of publication, see Releases.
The results of the full model can be found on our [comet page](https://www.comet.ml/bw4sz/deeplidar/2645e41bf83b47e68a313f3c933aff8a).
