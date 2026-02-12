# DFCCNet
![image](https://github.com/2226450890/DCCNet/blob/master/eight.jpg)
A dense flock of chickens counting model based on density map regression.
If you want to learn more about the method, read the article: https://doi.org/10.3390/ani13233729
## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Data Setup
Download Dense-Chicken Dataset from
OneDrive: [link](https://stuscaueducn-my.sharepoint.com/:u:/g/personal/3170062_stu_scau_edu_cn/ETT-vDigmvZBu6EgSRtSn0sBnNHLojY_tDmiVaoZteVP3g?e=rGa2yO) 

If you want to explore video based chicken counting, download these videos：
Baidu Netdisk: [link](https://pan.baidu.com/s/1CcNxEQ9XB2PJrxSBOHvpqw?pwd=crx3)

All references to data need to cite our article and indicate the source of the data.

## Evaluation
&emsp;1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from OneDrive: [link](https://stuscaueducn-my.sharepoint.com/:u:/g/personal/3170062_stu_scau_edu_cn/EV5-CSBgb2NPmBfWz_ks9woBKOb5vc42cW4BG6IfWQF4rQ?e=ydp6FM).  
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
## Citing
&emsp;Please cite this article if our code and data were helpful to you! https://doi.org/10.3390/ani13233729

&emsp;Further research results can be found at https://doi.org/10.1016/j.compag.2025.111172
