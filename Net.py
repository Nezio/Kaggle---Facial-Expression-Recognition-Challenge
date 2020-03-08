import tensorflow as tf
import numpy as np
from tqdm import tqdm

def Main():
    
    model = tf.keras.Sequential()

    # 32298 it [02:37, 205.64it/s]
    #print("Loading training data...")
    #train_data, train_labels = LoadData("train.csv")

    # 3589 it [00:17, 205.12it/s ]    
    print("Loading test data...")
    test_data, test_labels = LoadData("test.csv")

    '''filename = "test1.dat"
    fileobj = open(filename, mode='xb')
    test_data.tofile(fileobj)
    fileobj.close() '''
    
    
    print("start dat load")

    filename = "test1.dat"
    fileobj2 = open(filename, mode='rb')
    loaded_data = np.fromfile(fileobj2, dtype = np.float64)
    fileobj2.close()
    reshaped = np.reshape(loaded_data, (-1, 48*48))

    print("end dat load")

    print("All done!")


def LoadData(file):
    labels = []
    pixels = []

    # go trough rows of the csv file
    for line in tqdm(open(file)):
        row = line.split(',')
        labels.append(int(row[0]))
        pixels.append([int(pixel) for pixel in row[1].split()])

    # normalize data and save as a numpy array
    pixels = np.array(pixels) / 255.0
    labels = np.array(labels)

    return pixels, labels

if __name__ == "__main__":
    Main()