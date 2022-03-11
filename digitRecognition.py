from logging import exception
import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl
#Setting an HTTPS Context to fetch data from OpenML 
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

print('Before download')
x,y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print("After downlaod")
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_classes = len(classes)

x_train, x_test, y_train, y_test = tts(x, y, random_state = 9, test_size = 2500, train_size = 7500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

print('before clf')
clf = LR(solver = 'saga', multi_class='multinomial').fit(x_train_scaled, y_train)
print('after clf')
y_pred = clf.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
#Starting Camera Code (C123)

cap = cv2.VideoCapture(0)
while(True):
    try: 
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Drawing a box in the center of the video frame to make a location in which the readable number will have to be
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        #Creating the actual box
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        # Region of interest (point of interest)
        poi = gray[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right [0]]

        # using image function from PIL, we will convert the array pixels into a PIL Image
        img_pil = Image.fromarray(poi)
        img_bw = img_pil.convert('L')
        img_bw_resized = img_bw.resize((28, 28), Image.ANTIALIAS)
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 784)

        test_pred = clf.predict(test_sample)
        print("The Predicted Number Is...", test_pred)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as E: 
        print(E)
        

cap.release()
cv2.destroyAllWindows()