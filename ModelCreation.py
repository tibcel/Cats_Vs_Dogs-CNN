import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

IMG_SIZE = 100
DATASET_PATH = "PetImages"

TRAINING_DATA_FILENAME = "training_data.npy"

mappings = {}

def create_data(dataset_path, training_data_filename):
        
    training_data = []
      
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        print(f"Processing {dirpath}")
        if dirpath is not dataset_path:
            mappings[i-1] = (dirpath.split("/")[-1]) # adds Dog or Cat to the corresponding label
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(resized_img), i-1])
                except Exception as e:
                    pass
                
    np.random.shuffle(training_data)
    np.save(training_data_filename, training_data)
    
    with open("data.pkl", "wb") as d:
        pickle.dump(mappings, d)
        d.close()
        

def count_images(training_data):
    #The purpose of this function is to count how many dogs/cats are in the dataset to see if there is an unbalance
    cat_count = 0
    dog_count = 0
    
    for X, y in training_data:
        if y == 0:
            dog_count+=1
        elif y == 1:
            cat_count+=1
            
    print(f"There are {dog_count} dogs and {cat_count} cats")
    

def build_model(input_shape):
    
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu"))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu"))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    return model

def plot_history(history):
    
    fig, axs = plt.subplots(2)
    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    #create error/loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")
    
    plt.show()
    
#-----------------------------------------------------------------------------------------------------------------
            


if __name__=="__main__":
    
    create_data(DATASET_PATH, TRAINING_DATA_FILENAME)

    training_data = np.load(TRAINING_DATA_FILENAME, allow_pickle=True)
    
    count_images(training_data)
        
    X = [] #features list
    y = [] #label list
    
    for features, label in training_data:
        X.append(features)
        y.append(label)
        
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #make it compatible with tensorflow
    X = X/255.0  #to have values between 0->1
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    
    model = build_model(input_shape)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=5)
    plot_history(history)
    model.save("model")
































































                
    
                
    