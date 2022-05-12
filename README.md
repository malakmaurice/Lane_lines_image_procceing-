## Names:
- Abdelrahman Hany Gamal Bakr	1700773		sec 3
- Amr Mohamed Fatouh Ahmed 	1700948		sec 3
- Malak Mourice Henry		1701462		sec 5


## Usage:

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the pipeline from exe.bat (for windows)

for video or image without debugging 
```
exe.bat --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debug 
```bash
exe.bat --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```

### 3. Run the pipeline from shell.sh (for linux)
for video or image without debugging 
```
./shell.sh --<video|image> --no-debug <INPUT_PATH> <OUTPUT_PATH>
```
for video or image with debug 
```bash
./shell.sh --<video|image> --debug <INPUT_PATH> <OUTPUT_PATH>
```