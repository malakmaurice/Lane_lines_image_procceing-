import pickle

# Save classifier to pickle file
def save_classifier(pickle_file, svc, X_scaler, params):
    try:
        with open(pickle_file, "wb") as pfile:
            pickle.dump(
                {   "svc":svc, 
                    "scaler": X_scaler,
                    "color_space": params["color_space"],
                    "orient": params["orient"],
                    "pix_per_cell": params["pix_per_cell"],
                    "cell_per_block": params["cell_per_block"],
                    "hog_channel": params["hog_channel"],
                    "spatial_size": params["spatial_size"],
                    "hist_bins": params["hist_bins"],
                    "spatial_feat": params["spatial_feat"],
                    "hist_feat": params["hist_feat"],
                    "hog_feat": params["hog_feat"]
                },
                pfile)
    except Exception as e:
        print("Unable to save classifier to", pickle_file, ":", e)
        raise


# Load classifier from pickle file
def load_classifier(pickle_file):
    dist_pickle = pickle.load(open(pickle_file, "rb"))
    svc_data = {}
    svc_data["svc"]            = dist_pickle["svc"]
    svc_data["X_scaler"]       = dist_pickle["scaler"]
    svc_data["color_space"]    = dist_pickle["color_space"]
    svc_data["spatial_size"]   = dist_pickle["spatial_size"]
    svc_data["hist_bins"]      = dist_pickle["hist_bins"]
    svc_data["orient"]         = dist_pickle["orient"]
    svc_data["pix_per_cell"]   = dist_pickle["pix_per_cell"]
    svc_data["cell_per_block"] = dist_pickle["cell_per_block"]
    svc_data["hog_channel"]    = dist_pickle["hog_channel"]
    svc_data["spatial_feat"]   = dist_pickle["spatial_feat"]
    svc_data["hist_feat"]      = dist_pickle["hist_feat"]
    svc_data["hog_feat"]       = dist_pickle["hog_feat"]
    
    return svc_data