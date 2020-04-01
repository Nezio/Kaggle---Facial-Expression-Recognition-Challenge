import datetime
import shutil
import os
from random import randrange

import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from keras import metrics
from keras.utils import plot_model
from matplotlib import pyplot

import models
from tqdm import tqdm

################## CONFIG SECTION START ################################################################

# Filename for train and test data (without extension). If .dat file is present it will be used, 
# if not .csv file will be used and .dat file will be generated to be used in the next run. 
# This is to improve the load speeds as .dat file loads much faster with numpy array.
train_data_file = "data/train"
test_data_file = "data/test"

# Number of data samples to use for training (and validation). Set to 0 to use all the data.
train_subset_length = 4000

# Number of data samples to use for testing. Set to 0 to use all the data.
test_subset_length = 0

batch_size = 64
epochs = 4
validation_percentage = 0.2

# Model files to load (without extension). Model file (.json) and weights file (.h5) will be loaded.
# This will be ignored if left empty or "retrain" is set to "True".
model_files = "saved_models/model_test"

# If "retrain" is "True", training will be done regardless of whether the the weights file is provided or not.
# Training will always generate a weights file and save it to saved_models/model.json and saved_models/model.h5
retrain = True

# Whether to save wrongly classified images to folder for reviewing.
save_wrong_classifications = False

# Path where to save wrongly classified images if save_wrong_classifications is set to "True"
wrong_classifications_path = "data/images/wrong classifications"

# Number of images to save from wrong classifications.
wrong_classification_sample_size = 100

################## CONFIG SECTION END ##################################################################

emotions_mapping = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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

    # subset train data
    if (train_subset_length > 0):
        train_data = train_data[0:train_subset_length]
        train_labels = train_labels[0:train_subset_length]

    # reshape to fit Conv2D layer
    train_data = train_data.reshape(-1, 48, 48, 1)
    test_data = test_data.reshape(-1, 48, 48, 1)

    # save test data images
    #save_images(test_data)

    # get model by training or loading wights
    if (retrain or model_files == ""):
        # train the model
        model = models.baseline()

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])

        history = model.fit(train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=validation_percentage)

        # save the model
        save_model(model)

        # Plot training & validation loss values
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('Model loss')
        pyplot.ylabel('Loss')
        pyplot.xlabel('Epoch')
        pyplot.legend(['Train', 'Validation'], loc='upper left')
        pyplot.show()

        # Plot training & validation accuracy values
        pyplot.plot(history.history['categorical_accuracy'])
        pyplot.plot(history.history['val_categorical_accuracy'])
        pyplot.title('Model accuracy')
        pyplot.ylabel('Accuracy')
        pyplot.xlabel('Epoch')
        pyplot.legend(['Train', 'Validation'], loc='upper left')
        pyplot.show()
    else:
        # load the model
        model = load_model(model_files)

        # compile if needed
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

    # create nmodel vizualization image
    #plot_model(model, to_file='model.png')

    # subset test data
    if (test_subset_length > 0):
        test_data = test_data[0:test_subset_length]

    # predict outputs on test data
    print_log("Predicting answers on the test data...")
    prediction_np = model.predict(test_data, verbose=1)

    # format prediction and test_labels to an array of results
    predictions = [np.argmax(item) for item in prediction_np]
    correct_answers = [np.argmax(item) for item in test_labels]
    
    # calculate accuracy (prediction_correct is an array telling if prediction is correct, for each prediction)
    prediction_correct = [(prediction == correct_answer) for prediction, correct_answer in zip(predictions, correct_answers)]
    accuracy = np.mean(prediction_correct)

    print_log("Accuracy on the test data is: {accuracy}".format(accuracy=accuracy))

    # save a sample of incorrect classifications
    if(save_wrong_classifications):
        save_wrong_classification_sample(test_data, predictions, correct_answers)
        print_log('Saved {image_number} wrongly classified images to "{folder}".'.format(image_number=wrong_classification_sample_size,folder=wrong_classifications_path))

    print_log("All done!")



##############################################################################################################################################
##############################################################################################################################################



def load_data(file):
    data_file = file
    labels_file = file + "_labels"

    # if both data and labels files exist as .dat, load from there
    if (os.path.exists(data_file + ".dat") and os.path.exists(labels_file + ".dat")):
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

def save_model(model):
    # save json
    model_json = model.to_json()
    with open("saved_models/model.json", "w") as json_file:
        json_file.write(model_json)

    # save the weights
    model.save_weights("saved_models/model.h5")
    print_log("Model saved to disk.")

def load_model(model_path):
    # read model from json
    model_json_file = open(model_path + ".json", 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)

    # load weights
    model.load_weights(model_path + ".h5")

    print_log('Model loaded from "{model_path}"'.format(model_path=model_path))

    return model

def save_wrong_classification_sample(images, predictions, correct_answers):
    '''
        Saves a sample of wrongly classified images as labeled image files.

        Parameters:
        images: Numpy array of images. The data that prediction was ran on.
        predictions: Numpy array of predictions (array of integer values for classes).
        correct_answeres: Numpy array of actual correct classifications.
    '''
    data_length = images.shape[0]
    already_selected = []

    # clear the folder of previous images and recreate it
    if (os.path.exists(wrong_classifications_path)):
        shutil.rmtree(wrong_classifications_path)

    os.mkdir(wrong_classifications_path)

    for i in range(wrong_classification_sample_size):
        random_index = randrange(data_length)

        # skip already saved samples and skip correct predictions
        while (random_index in already_selected or predictions[random_index] == correct_answers[random_index]):            
            random_index = randrange(data_length)

        already_selected.append(random_index)

        # save the image
        image = images[random_index]
        image = image.reshape(48, 48)
        # 1 - Predicted=Happy, Actual=Neutral
        image_name = str(i+1) + " - Predicted=" + emotions_mapping[predictions[random_index]] + ", Actual=" + emotions_mapping[correct_answers[random_index]]
        image_path = wrong_classifications_path + "/" + image_name + '.png'
        pyplot.imsave(image_path, image, cmap='gray')

def save_images(images):
    '''
        Save provided Nympy array of images as .png.
    '''
    for i in tqdm(range(images.shape[0])):
        image = images[i]
        image = image.reshape(48,48)
        image_path = 'data/images/test data/' + str(i) + '.png'
        pyplot.imsave(image_path, image, cmap='gray')





if __name__ == "__main__":
    main()