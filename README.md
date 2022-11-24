# Learning Trajectory Prediction for Multiple Vehicles at Intersections using State and Intention Information

## Abstract
Traditional approaches to prediction of future trajectory of road agents rely on knowing information about their past trajectory. This work rather relies only on having knowledge of the current state and intended direction to make predictions for multiple vehicles at intersections. Furthermore, message passing of this information between the vehicles provides each one of them a more holistic overview of the environment allowing for a more informed prediction. This approach can further be extended to additionally control the multiple vehicles to drive towards desired paths  by manipulating their intention. Experimental results demonstrate the robustness of our approach both in terms of trajectory prediction and vehicle control at intersections.

![image](images/overview.png)

## Results
* Controlling vehicles without message passing (baseline) [[video]](https://drive.google.com/file/d/1AdF94g7Gd6ytB8gMKFR0qGHDBQH6qtnC/view?usp=share_link)
* Controlling vehicles with message passing. [[video]](https://drive.google.com/file/d/1V74KRbbgGNnIJ3vWwqs3vQ9wO5Y01D2S/view?usp=share_link)

## Setup

### 1) Packages Install

``` bash
conda create -n mvn python=3.7
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
conda install matplotlib
conda install -c conda-forge cvxpy
conda install -c anaconda lxml
```

### 2) Software Install
* CARLA: 0.9.10
* SUMO: 1.13.0

## Run the Inference code

### 1) Put the inference code and the map in place

First navigate to the directory where Carla installed, then copy the directories and the scripts in this repository and put them in the correct place as the following image shown. 

![image](images/setup.png)

*Red: the existing script or directory needs to be replaced*

*Green: a new created script or directory*


### 2) Activate our map in CARLA

First execute carla.exe or carla.sh, then run the following commands: 
```bash
conda activate mvn

cd ${Carla_folder}/PythonAPI/util

python config.py -x ../../Co-Simulation/Sumo/sumo_files/map/map_15m.xodr
```
Now the intersection scenario should be already ativated!

### 3) Run the inference code
If the above steps all work properly, we can finally run the inference code to control the vehicles at this intersection, just run the following commands:
```bash
cd ${Carla_folder}/Co-Simulation/Sumo

python run_synchronization.py  sumo_files/sumocfg/09-11-15-30-00400-0.09-val_10m_35m-7.sumocfg  --tls-manager carla  --sumo-gui  --step-length 0.1
```
