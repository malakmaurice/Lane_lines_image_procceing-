# SIMPLE PERCEPTION STACK FOR SELF-DRIVING CARS

## Table of Contents
- [Names](#names)
- [Notes](#notes)
- [Usage](#usage)
    - [Install Dependencies](#install-dependencies)
    - [Dectect Lane](#detect-lane)
        - [For windows](#for-windows)
        - [For linux](#for-linux)
    - [Detect Cars YOLO](#detect-cars-yolo)
        - [For windows](#for-windows-1)
        - [For linux](#for-linux-1)
    - [Detect Cars SVM](#detect-cars-svm)
        - [For windows](#for-windows-2)
        - [For linux](#for-linux-2)

## Names
- Abdelrahman Hany Gamal Bakr	1700773		sec 3
- Amr Mohamed Fatouh Ahmed 	1700948		sec 3
- Malak Mourice Henry		1701462		sec 5


## Notes
- A link to our repo can be found <a href="https://github.com/malakmaurice/Lane_lines_image_procceing-">here</a>
- Output <b>images</b> and <b>videos</b> can be viewed <a href="https://drive.google.com/drive/folders/1jsCUQx13wl2iNkYaZXzwvM8nG9x9shYV?usp=sharing">here</a>
- YOLO weights are not uploaded, so you have to download them from <a href="https://pjreddie.com/darknet/yolo/">here</a> (make sure that you download the weights corresponding to YOLOv3-416)
- the config file for YOLO can be found in  <b><i>phase2/YOLO/yolov3.cfg</i></b>
- the labels file for YOLO can be found in  <b><i>phase2/YOLO/coco.names</i></b>

## Usage

### Install Dependencies
```
pip install -r requirements.txt
```

### Detect Lane

#### For windows
for video or image with or without debugging 
```
exe.bat --detect-lane --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```

#### For linux
for video or image with or without debugging 
```
./shell.sh --detect-lane --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```

<img src="output_images/out8.jpg">

<br>
<hr>
<br>

### Detect Cars YOLO
#### For windows
for video or image
```
exe.bat --detect-cars-yolo <weights_path> <config_path> <labels_path> --<video|image> <INPUT_PATH> <OUTPUT_PATH>
```
#### For linux
for video or image
```
./shell --detect-cars-yolo <weights_path> <config_path> <labels_path> --<video|image> <INPUT_PATH> <OUTPUT_PATH>
```
<img src="output_images/phase2/YOLO/out8.jpg">

<br>
<hr>
<br>

### Detect Cars SVM

#### For windows
for video or image with or without debugging 
```
exe.bat --detect-cars-svm --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```
#### For linux
for video or image with or without debugging 
```
./shell.sh --detect-cars-svm --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```

<img src="output_images/phase2/SVM/out8.jpg">

<br>
<hr>
<br>

### Train SVM Model

#### For windows
```
exe.bat --train-svm-model <dataset_path>
```

#### For linux
```
./shell.sh --train-svm-model <dataset_path>
```