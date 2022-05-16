import glob

# Returns two lists: cars and notcars list of image files.
def load_data_set(dir):
    cars = glob.glob(dir+"/vehicles/GTI_Far/*.png")
    cars += glob.glob(dir+"/vehicles/GTI_MiddleClose/*.png")
    cars += glob.glob(dir+"/vehicles/GTI_Left/*.png")
    cars += glob.glob(dir+"/vehicles/GTI_Right/*.png")
    cars += glob.glob(dir+"/vehicles/KITTI_extracted/*.png")
    noncars = glob.glob(dir+"/non-vehicles/Extras/*.png")
    noncars += glob.glob(dir+"/non-vehicles/GTI/*.png")
    return cars, noncars