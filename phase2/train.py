import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from phase2.load_data_set import load_data_set
from phase2.get_extracted_features_from_data import get_extracted_features_from_data
from phase2.save_load_pickle import save_classifier


# HOG parameters
params = {}
params['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 9  # HOG orientations
params['pix_per_cell'] = 8 # HOG pixels per cell
params['cell_per_block'] = 2 # HOG cells per block
params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
params['spatial_size'] = (32, 32) # Spatial binning dimensions
params['hist_bins'] = 32    # Number of histogram bins
params['spatial_feat'] = True # Spatial features on or off
params['hist_feat'] = True # Histogram features on or off
params['hog_feat'] = True # HOG features on or off


def train(data_path, output_path, debug):
    
    if(debug):
        print("Loading dataset...")
    else:
        print("Extracting data then train them...")
    
    cars, noncars = load_data_set(data_path)

    if(debug):
        print("Extraxting features from data...")
    t1=time.time()
    cars_feats, noncar_feats = get_extracted_features_from_data(params, cars, noncars)
    t2 = time.time()
    if(debug):
        print("Extracting features done in: ", round(t2-t1, 2), "seconds")

    #---------------------------------------------------------------------------------

    # PREPARE DATA FOR TRAINING
    if(debug):
        print("Preparing data for training...")

    X = np.vstack((cars_feats, noncar_feats)).astype(np.float64)

    X_scaler = StandardScaler().fit(X)

    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(cars)), np.zeros(len(noncars))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    if(debug):
        print('Feature vector length:', len(X_train[0]))
    
    #---------------------------------------------------------------------------------

    svc_classifier = LinearSVC()

    if(debug):
        print("Training SVC...")
    t1=time.time()
    svc_classifier.fit(X_train, y_train)
    t2 = time.time()
    if(debug):
        print("Training done in: ", round(t2-t1, 2), "seconds")

    score = svc_classifier.score(X_test, y_test)
    if(debug):
        print("Test Accuracy of SVC = ", round(score, 4))

    t1=time.time()
    n_predict = 100
    if(debug):
        print("SVC predicts: ", svc_classifier.predict(X_test[0:n_predict]))
        print("For these",n_predict, "labels: ", y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t1, 5), "Seconds to predict", n_predict,"labels with SVC")

    if(debug):
        print("Saving classifier to pickle file...")
    save_classifier(output_path, svc_classifier, X_scaler, params)
    if(debug):
        print("Classifier saved in: ", output_path)
