import matplotlib.image as mpimg
from phase2.hog_utils import single_img_features

# get the HOG features of the images in a list
def extract_features(imgs, params):
                     
    color_space    = params['color_space']
    spatial_size   = params['spatial_size']
    hist_bins      = params['hist_bins']
    orient         = params['orient']
    pix_per_cell   = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel    = params['hog_channel']
    spatial_feat   = params['spatial_feat']
    hist_feat      = params['hist_feat']
    hog_feat       = params['hog_feat']
    
    features = []
    
    for file in imgs:
        img = mpimg.imread(file)
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        features.append(img_features)

    return features


# extracts features from images of cars and non-cars
def get_extracted_features_from_data(params, cars, noncars):    
    car_features = extract_features(cars, params)
    noncar_features = extract_features(noncars, params)
    
    assert(len(car_features) == len(cars))
    assert(len(noncar_features) == len(noncars))

    return car_features, noncar_features