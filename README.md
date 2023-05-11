# DCCNet
![image](https://github.com/2226450890/DCCNet/blob/master/eight.jpg)
A dense flock of chickens counting model based on density map regression.

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Data Setup
Download Dense-Chicken Dataset from
Baidu Disk: [link](http://pan.baidu.com/s/1nuAYslz)  

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from Baidu Disk: [link](http://pan.baidu.com/s/1nuAYslz).  
&emsp;2. To run code quickly, we have to set up the following directory structure in our project directory.
    
```
DCCNet                                           # Project folder. Typically we run our code from this folder.
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

# Training
&emsp;1. Configure the files required to run, and modify the root path in the "train.py" based on your dataset location.   
&emsp;2. Run train.py:
```
python train.py
```  
