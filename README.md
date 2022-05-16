# NAMES
- Abdelrahman Hany Gamal Bakr	1700773		sec 3
- Amr Mohamed Fatouh Ahmed 	1700948		sec 3
- Malak Mourice Henry		1701462		sec 5


# USAGE

## Install Dependencies
```
pip install -r requirements.txt
```

## Detect Lane

### For windows
for video or image without debugging 
```
exe.bat --detect-lane --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debugging 
```bash
exe.bat --detect-lane --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```
### For linux
for video or image without debugging 
```
./shell.sh --detect-lane --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debugging 
```bash
./shell.sh --detect-lane --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```


## Detect Cars
### For windows
for video or image without debugging 
```
exe.bat --detect-cars --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debugging 
```bash
exe.bat --detect-cars --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```
### For linux
for video or image without debugging 
```
./shell.sh --detect-cars --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debugging 
```bash
./shell.sh --detect-cars --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```