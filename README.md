# DFCCNet
![image](https://github.com/2226450890/DCCNet/blob/master/eight.jpg)
A dense flock of chickens counting model based on density map regression.

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Data Setup
Download Dense-Chicken Dataset from
aliyun drive: [link](https://www.aliyundrive.com/s/3sDZtJZcSsZ) 

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from aliyun drive: [link](https://www.aliyundrive.com/s/vR1iU2vP1od).  
&emsp;2. To run code quickly, we have to set up the following directory structure in our project directory.
    
```
DFCCNet                                           # Project folder. Typically we run our code from this folder.
│   
│───chicken                                      # Folder where we save run-related files.
│   │───conv                                     # Folder that contains the feature convolution kernels 
│   │───images                                   # Folder where we save images.
│   │───chicken_annotation.json                  # Annotation file. 
│   └───Train_Test_Val_chicken.json              # File that divides the dataset
│                               
│───result                                       # Folder where we save pretrained model.
│   │───backbone.pth
│   └───generate_density.pth     
│                               
│───density_show.py                              # Density map visualization file.
│───generate_conv.py                             # File that generates feature convolution kernels.
│───model.py                                     # File of the model.
│───train.py                                     # File with the code to train the model.
│───test.py                                      # File with the code to test the model.
│───utils.py                                     # File contains functions that other files need to use
└───README.md                                    # Readme of the project.
```
&emsp;3. Evaluate the model
```
python test.py
```  

## Training
&emsp;1. Configure the files required to run, and modify the root path in the "train.py" based on your dataset location.  
&emsp;2. Generates feature convolution kernels:
```
python generate_conv.py
```  
&emsp;3. Run train.py:
```
python train.py
```  
