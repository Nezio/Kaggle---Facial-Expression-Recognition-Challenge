# Filename for train and test data (without extension). If .dat file is present it will be used, 
# if not .csv file will be used and .dat file will be generated to be used in the next run. 
# This is to improve the load speeds as .dat file loads much faster with numpy array.
train_data_file = "data/train"
test_data_file = "data/test"

# Number of data samples to use for training (and validation). Set to 0 to use all the data.
train_subset_length = 500

# Number of data samples to use for testing. Set to 0 to use all the data.
test_subset_length = 100

batch_size = 128
epochs = 2
validation_percentage = 0.2

# Model files to load (without extension). Model file (.json) and weights file (.h5) will be loaded.
# This will be ignored if left empty or "retrain" is set to "True".
model_files = "saved_models/model_test"

# If "retrain" is "True", training will be done regardless of whether the the weights file is provided or not.
# Training will always generate a weights file and save it to saved_models/model.json and saved_models/model.h5
retrain = True

# Whether to save wrongly classified images to folder for reviewing.
save_wrong_classifications = True

# Number of images to save from wrong classifications.
wrong_classification_sample_size = 10

# Folder to which output will be saved (Model, weights, plots, wrong classification sample images, configuration...).
# Folder name will be the current datetime.
output_root_path = "output"

# Add a nickname to the genereated output folder. Folder will be named "<datetime> - nickname".
output_folder_nickname = ""

# Should model vizualization chart be created?
create_model_vizualization = True

# A string representing which model design to use. Select one from models.py.
model = "baseline"