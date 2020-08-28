from tensorflow import keras
import ModelCreation
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import tkinter as tk
from tkinter import filedialog, Text

PATH_TO_MODEL = "model"
IMG_SIZE = ModelCreation.IMG_SIZE

model = keras.models.load_model(PATH_TO_MODEL) #load model generated from ModelCreation

with open("data.pkl", "rb") as d:
    mappings = pickle.load(d)
    d.close()


def make_prediction(path_to_img):
    
    X = [] #create empty list to append to
    
    img = cv2.imread(os.path.join(path_to_img), cv2.IMREAD_GRAYSCALE)    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #getting the image to the desired size
    X.append(img)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #resizing it to be compatible with the model
    X = X/255.0
    
    prediction = model.predict(X)
    print(prediction)
    
    return mappings[np.argmax(prediction)]
    
#---------------------------------------------------------------------------------------------------------

root = tk.Tk()

def add_file():
    filename = filedialog.askopenfilename(initialdir = "/home/bibi/Desktop/Programare/Proiecte Personale/Cats_vs_Dogs", title = "Select Image")
    prediction = make_prediction(filename)
    
    label = tk.Label(root, text="I think that you've selected a "+prediction)
    label.pack()


open_file_button = tk.Button(root, text="Select file", command=add_file)
open_file_button.pack()





root.mainloop()
   



