import numpy as np
import cv2
from phase2.hog_utils import convert_rgb_color, get_hog_features, bin_spatial, color_hist


# Apply a sliding window for a given search region, then extract its features, then apply the trained classifier to the features.
def find_cars(img, ystart, ystop, scale, svc, X_scaler, params, cells_per_step):

    # Define HOG parameters
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
    
    assert(hog_channel == 'ALL')
    assert(spatial_feat == True)
    assert(hist_feat == True)
    assert(hog_feat == True)
    
    
    img_tosearch = img[ystart:ystop,:,:]
    
    
    ctrans_tosearch = convert_rgb_color(img_tosearch, conv=color_space)
    
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    
    ch1 = ctrans_tosearch[:,:,0]    # Y channel
    ch2 = ctrans_tosearch[:,:,1]    # Cr channel
    ch3 = ctrans_tosearch[:,:,2]    # Cb channel

    
    nxcells = (ch1.shape[1] // pix_per_cell) - cell_per_block + 3
    nycells = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2  # Unused: number of features per block
    
    window = 64 # size of one side of the window in pixels
    ncells_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxcells - ncells_per_window) // cells_per_step
    nysteps = (nycells - ncells_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Apply sliding window
    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this window
            hog_feat1 = hog1[ypos:ypos+ncells_per_window, xpos:xpos+ncells_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+ncells_per_window, xpos:xpos+ncells_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+ncells_per_window, xpos:xpos+ncells_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # top-left corner of the window
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the part of the image covered by the current window
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction (Standardize features by removing the mean and scaling to unit variance)
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            
              
            test_prediction = svc.predict(test_features)
            
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # each bounding box contains the coordinates of the top-left corner and the bottom-right corner
                box = [(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)]
                bboxes.append(box)
                
    return bboxes



# iterate find_cars() over all multiple scales and return the list of bounding boxes for all scales
def find_cars_multiscale(img, svc_data, y_start, y_stop, scale, cells_per_step):
    
    assert(np.max(img) <= 1)
    assert(len(y_start) == len(y_stop) == len(scale))

    svc = svc_data['svc']
    X_scaler = svc_data['X_scaler']

    bboxes = []
    
    for i in range(len(y_start)):
        boxes = find_cars(img, y_start[i], y_stop[i], scale[i], svc, X_scaler, svc_data, cells_per_step)
        if len(boxes):
            bboxes.extend(boxes)

    return bboxes
