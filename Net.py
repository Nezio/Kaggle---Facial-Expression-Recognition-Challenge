import tensorflow as tf
import numpy as np
from tqdm import tqdm
from os import path
import datetime

def Main():

    ################## CONFIG SECTION ################################################################

    # Filename for train and test data (without extension). If .dat file is present it will be used, 
    # if not .csv file will be used and .dat file will be generated to be used in the next run. 
    # This is to improve the load speeds as .dat file loads faster with numpy array.
    train_data_file = "train"
    test_data_file = "test"

    ################## CONFIG SECTION ends; don't change variables past this point ##################

    # load train data
    # if loaded from .csv: 32298 it [02:37, 205.64it/s]
    train_data, train_labels = LoadData(train_data_file)
    
    # load test data
    # if loaded from .csv: 3589 it [00:17, 205.12it/s]
    test_data, test_labels = LoadData(test_data_file)
    

    

    PrintLog("All done!")

def LoadData(file):
    data_file = file
    labels_file = file + "_labels"

    # if both data and labels files exist as .dat, load from there
    if (path.exists(data_file + ".dat") and path.exists(labels_file + ".dat")):
        # load from .dat
        data = LoadDataFromDAT(data_file, np.float64)
        labels = LoadDataFromDAT(labels_file, np.int32)

        # reshape
        data = np.reshape(data, (-1, 48*48))
    else:
        # load from .csv
        data, labels = LoadDataFromCSV(data_file)

        # save to .dat
        SaveDataToDAT(data_file, data)
        SaveDataToDAT(labels_file, labels)

    return data, labels

def LoadDataFromCSV(file):
    """
        Returns numpy arrays for pixels and labels data of shape (n, 1) and (n, 48x48) respectively, where n is number of data samples.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".csv"):
        file = file + ".csv"

    labels = []
    pixels = []

    PrintLog('Started loading data from "{file}".'.format(file=file))

    # go trough rows of the csv file; print progress with "tqdm"
    for line in tqdm(open(file)):
        row = line.split(',')
        labels.append(int(row[0]))
        pixels.append([int(pixel) for pixel in row[1].split()])

    # normalize data and save as a numpy array
    pixels = np.array(pixels) / 255.0
    labels = np.array(labels)

    PrintLog('Done loading data from "{file}".'.format(file=file))

    return pixels, labels

def LoadDataFromDAT(file, data_type):
    """
        Returns numpy array of shape (n, 1) where "n" is number of data samples.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".dat"):
        file = file + ".dat"

    file_object = open(file, mode='rb')
    loaded_data = np.fromfile(file_object, dtype = data_type)
    file_object.close()

    PrintLog('Data loaded from "{file}".'.format(file=file))

    return loaded_data

def SaveDataToDAT(file, data):
    """
        Saves provided data to file. Data must be a numpy array.
    """

    # add .dat extension if needed
    if (len(file) < 4 or file[-4:] != ".dat"):
        file = file + ".dat"

    PrintLog('Started saving data to "{file}".'.format(file=file))

    file_object = open(file, mode='wb')
    data.tofile(file_object)
    file_object.close()

    PrintLog('Done saving data to "{file}".'.format(file=file))

def PrintLog(text):
    print("[" + datetime.datetime.now().strftime("%H:%M:%S") + "]", end="")
    print(" " + text)




if __name__ == "__main__":
    Main()