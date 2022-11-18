from flask import Flask
from flask import request
import base64
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import argmax
from keras.models import load_model
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['POST'])
def handle_request():
  imgName = request.form['imageName']
  imgBase64 = request.form['imageBase']
  save_image(imgName, imgBase64)
  return "Image Upload Successfull"

# save image to directory
def save_image(imageName, imageBase64):
  filename, image = preprocess_image(imageBase64)
  print(filename)
  curDirectory = os.getcwd()
  filePath = os.path.join(curDirectory, str(filename))
  if not os.path.exists(filePath):
    os.makedirs(filePath)
    
  cv2.imwrite(filePath + '/'+ imageName, image)
  return filename
	
# decode base64 array to image and correct image alignment
def preprocess_image(imageBase64):
  encoded_data = imageBase64
  nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  fname = test_image(img)
  return fname, img

# perform image processing
def load_image(img):  
  plt.imshow(img)
  plt.show()
  rgb_planes = cv2.split(img)
  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((70,70), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
  kernel = np.ones((5, 5), np.uint8)
  result = cv2.merge(result_planes)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
  result = cv2.erode(result, kernel, iterations=3)
  result = cv2.dilate(result, kernel, iterations=3)
  (_, im_bw) = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  mask = im_bw < 255
  coords = np.argwhere(mask)
  x0, y0 = coords.min(axis=0)
  x1, y1 = coords.max(axis=0) + 1
  cropped = im_bw[x0:x1, y0:y1]
  resized = cv2.resize(cropped, (18, 18), interpolation = cv2.INTER_AREA)
  resized = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255])
  resized = cv2.bitwise_not(resized)
  plt.imshow(resized, cmap="gray")
  plt.show()
  resized = resized.reshape(1, 28, 28, 1)
  resized = resized.astype('float32')
  resized = resized / 255.0
  return resized

# load an image and predict the class
def test_image(img):
  img = load_image(img)
  # load model
  model = load_model('ImageClassifier.h5')
  # predict the class
  predict_value = model.predict(img)
  digit = argmax(predict_value)
  print("Predicted Digit is ", digit)
  return digit
 
app.run()
# if __name__ == "__main__":
#   app.run(host='0.0.0.0', port=5000, debug=True)