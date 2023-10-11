# Automatic Detection Tool

Automated determination of the projectile velocity in impact tests

## Install

It is recommended to use conda to manage the Python environment.

[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 

Create a standalone python environment

```shell
conda create --name ADT python=3.9
conda activate ADT
```

Install the required python libraries

```shell
pip install PyQt6 roboflow matplotlib opencv-python roboflow
```

Download file

```shell
git clone https://github.com/HerrDanke/AutomaticDetectionTool.git
cd AutomaticDetectionTool
```

run

```shell
python main.py
```

**If you want to delete this python environment**

```shell
conda env remove --name ADT
```

## Instruction

- The default token in the tool is the API Key for my roboflow account, calling the model I deployed on roboflow.
- Please do not include special characters in the path. If there is an error, check the path.

<img src="https://raw.githubusercontent.com/HerrDanke/image/main/image-20231010215533445.png" alt="image-20231010215533445" style="zoom:67%;" />

The picture above is my user interface.

During the operation 3 folders will be generated under the save path selected by the user:

- VideoFrames -- After loading the video click on the button "Get Frames" and the folder containing the video frames will be saved in this folder.
- DetectFrames -- After loading the video and the model, click on the button "Start Prediction" and the folder containing the video frame detection results will be saved in this folder.
- DetectVideo -- Click on the button "write2video" and select a folder containing video frames. The generated video will be saved in this folder.

## Training

I have also provided training files. If you want to train your own model, you can refer to my training files.

First you need to install the notebook and YOLOv8 support libraries.

```
pip install notebook
pip install ultralytics
```

I manage my datasets through [Roboflow](https://roboflow.com/) and you can do the same with me.

I also recommend that you use [colab](https://colab.research.google.com/) to train the model. You can open this *ipynb* file in colab.

Here's some official documentation for roboflow, where you can [train](https://github.com/roboflow/notebooks) and [host](https://docs.roboflow.com/deploy/hosted-api/object-detection) your models.