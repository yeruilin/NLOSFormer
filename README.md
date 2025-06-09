# NLOSFormer
This is the public implementation for "Thermal Non-Line-of-Sight Imaging through Rough Surfaces".

## Install

First, create a new virtual environment.

`conda create -n NLOSFormer python==3.10`

`conda activate NLOSFormer`

Then, install all the packages required.

`pip install -r requirements.txt`

## Demo

The pretrained path is placed under "pth/", and the data are placed under "data/". You should unzip the .zip file under "data/" and place it under the directory "data/squat/".

Then, you can demonstrate thermal NLOS imaging of a dynamic target by `python demo.py`. Then you can see a video of the reconstructed target under "results/".

More results will be presented after the paper review.