## Digit-Classifier
Digit Classifier is a basic android application integrated with a flask server.<br>
The user can use the application to click images of handwritten digits and upload the clicked image to the server.<br>
The server receives the image array in a post request.<br>
The image array is then converted back into the original image.<br>
The server then imports a CNN model to classify the input digit for 0-9, preprocesses the input image to load into the model, and then stores the image in the folder based on the model output. <br>
The classifier model is trained on MNIST dataset and has ~99.2% accuracy on test data. <br>
You can load the trained model 'DigitClassifier.h5' or build a new model using 'model.py'. <br>

### Instructions to setup
Use command 'python index.py' in cmd terminal to run the flask local server<br>
Copy the localhost IP address from terminal<br>
Open 'DigitClassifier' project in android studio<br>
Replace the BASE_URL string value in the file APIContract.java with the localhost IP address from flask server. <br>
Run the android application<br>

### Important note regarding model training and loading
Library versions used during model training should be same as the ones used during model loading, otherwise the server-side code may not work as intended. <br>
To solve this issue, re-train the model in the system where server-side code will be executed. <br>
