import tensorflow as tf
import numpy as np
import datetime
import models
from tqdm import tqdm
from os import path

################## CONFIG SECTION START ################################################################

# Filename for train and test data (without extension). If .dat file is present it will be used, 
# if not .csv file will be used and .dat file will be generated to be used in the next run. 
# This is to improve the load speeds as .dat file loads faster with numpy array.
train_data_file = "data/train"
test_data_file = "data/test"

# Number of data samples to use for training. Set to 0 to use all data.
subset_length = 2000

batch_size = 128
epochs = 3

# Weights file to load. This will be ignored if "retrain" is "true" or left empty.
weights_file = ""

# If "retrain" is "true", training will be done regardless of whether the the weights file is provided or not.
# Training will always generate a weights file and prompt to save it.
retrain = "true"

################## CONFIG SECTION END ##################################################################

def main():

    # load train data
    # if loaded from .csv: 32298 it [02:37, 205.64it/s]
    train_data, train_labels = load_data(train_data_file)
    
    # load test data
    # if loaded from .csv: 3589 it [00:17, 205.12it/s]
    test_data, test_labels = load_data(test_data_file)
    
    # one-hot encode labels
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    # subset data
    if (subset_length > 0):
        train_data = train_data[0:subset_length]
        train_labels = train_labels[0:subset_length]

    # reshape to fit Conv2D layer
    train_data = train_data.reshape(-1, 48, 48, 1)

    # get model by training or loading wights
    if (retrain or weights_file == ""):
        # train the model
        model = models.baseline_model()

        model.fit(train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.1111)

        # TODO: save prompt

    else:
        # load weights
        
    # predict outputs on test data
    
    


    print_log("All done!")


def load_data(file):
    data_file = file
    labels_file = file + "_labels"

    # if both data and labels files exist as .dat, load from there
    if (path.exists(data_file + ".dat") and path.exists(labels_file + ".dat")):
        # load from .dat
        data = load_data_from_dat(data_file, np.float64)
        labels = load_data_from_dat(labels_file, np.int32)

        # reshape
        data = np.reshape(data, (-1, 48*48))
    else:
        # load from .csv
        data, labels = load_data_from_csv(data_file)

        # save to .dat
        save_data_to_dat(data_file, data)
        save_data_to_dat(labels_file, labels)

    return data, labels

def load_data_from_csv(file):
    """
        Returns numpy arrays for pixels and labels data of shape (n, 1) and (n, 48x48) respectively, where n is number of data samples.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".csv"):
        file = file + ".csv"

    labels = []
    pixels = []

    print_log('Started loading data from "{file}".'.format(file=file))

    # go trough rows of the csv file; print progress with "tqdm"
    for line in tqdm(open(file)):
        row = line.split(',')
        labels.append(int(row[0]))
        pixels.append([int(pixel) for pixel in row[1].split()])

    # normalize data and save as a numpy array
    pixels = np.array(pixels) / 255.0
    labels = np.array(labels)

    print_log('Done loading data from "{file}".'.format(file=file))

    return pixels, labels

def load_data_from_dat(file, data_type):
    """
        Returns numpy array of shape (n, 1) where "n" is number of data samples.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".dat"):
        file = file + ".dat"

    file_object = open(file, mode='rb')
    loaded_data = np.fromfile(file_object, dtype = data_type)
    file_object.close()

    print_log('Data loaded from "{file}".'.format(file=file))

    return loaded_data

def save_data_to_dat(file, data):
    """
        Saves provided data to file. Data must be a numpy array.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".dat"):
        file = file + ".dat"

    print_log('Started saving data to "{file}".'.format(file=file))

    file_object = open(file, mode='wb')
    data.tofile(file_object)
    file_object.close()

    print_log('Done saving data to "{file}".'.format(file=file))

def print_log(text):
    print("[" + datetime.datetime.now().strftime("%H:%M:%S") + "]", end="")
    print(" " + text)

def one_hot_encode(input_array):
    '''
        Convert provided two-dimensional numpy array to one-hot encoding.
    '''

    # shape of the new one-hot encoded numpy array
    shape = (input_array.size, input_array.max() + 1)

    # fill the shape with zeros
    one_hot_array = np.zeros(shape)

    # generate numpy array ranging from 0 to the number of data samples; this will be used as indexer for the one-hot array
    rows = np.arange(input_array.size)

    # first index selects row (one by one, from 0 to number of samples), second index selects column matching the label value
    one_hot_array[rows, input_array] = 1

    return one_hot_array






if __name__ == "__main__":
    main()