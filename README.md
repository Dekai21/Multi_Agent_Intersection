# Learning Navigation for Multiple Vehicles at Intersections

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

![image](setup.png)

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
