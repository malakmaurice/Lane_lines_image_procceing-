# NAMES
- Abdelrahman Hany Gamal Bakr	1700773		sec 3
- Amr Mohamed Fatouh Ahmed 	1700948		sec 3
- Malak Mourice Henry		1701462		sec 5

# REPO LINK
<a href="https://github.com/malakmaurice/Lane_lines_image_procceing-">Repo Link</a>

# NOTES
- Output <b>images</b> and <b>videos</b> can be viewed <a href="https://drive.google.com/drive/folders/1jsCUQx13wl2iNkYaZXzwvM8nG9x9shYV?usp=sharing">here</a>
- YOLO weights are not uploaded, so you have to download it from <a href="https://pjreddie.com/darknet/yolo/">here</a> (make sure that you download the weights corresponding to YOLOv3-416)

# USAGE

## Install Dependencies
```
pip install -r requirements.txt
```

## Detect Lane

### For windows
for video or image with or without debugging 
```
exe.bat --detect-lane --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```

### For linux
for video or image with or without debugging 
```
./shell.sh --detect-lane --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```

## Detect Cars YOLO
### For windows
for video or image
```
exe.bat --detect-cars-yolo <weights_path> <config_path> <labels_path> --<video|image> <INPUT_PATH> <OUTPUT_PATH>
```
### For linux
for video or image
```
./shell --detect-cars-yolo <weights_path> <config_path> <labels_path> --<video|image> <INPUT_PATH> <OUTPUT_PATH>
```

## Detect Cars SVM

### For windows
for video or image with or without debugging 
```
exe.bat --detect-cars-svm --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```
### For linux
for video or image with or without debugging 
```
./shell.sh --detect-cars-svm --<video|image> --<debug|no-debug> <INPUT_PATH> <OUTPUT_PATH>
```