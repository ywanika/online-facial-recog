from flask import Flask, render_template, request, redirect
import keras
from keras.models import load_model
import numpy as np
import cv2
import os
import random
from passlib.hash import pbkdf2_sha256

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/static"
app.config["SECRET_KEY"] = "gguu"

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #load face classifier
model = load_model ("famousFace.h5") #load the model to identify the faces

def random_name_generator():
    """
        Create a random name using 16 numbers and return it
    """
    fileName = ""
    for x in range (16):
        fileName+= (chr(random.randint(48,57)))
    return fileName + ".png"

def recog_face(user_image):
    """
        This funtion identifies the faces and adds the names to the images under the face
        Parameters: the image passed by the user
        Output: an edited image with the names in the static folder, a path to the images, a list of the names
    """
    names = [] #list of names in image

    im_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY) #convert image to gray

    faces = face_cascade.detectMultiScale(im_gray, 1.3, 5) #detect the faces
    for x in faces: #loop through the faces
        #x is a list: [x, y, width, hieght]
        roi = im_gray[x[1] : x[1]+x[3], x[0] : x[0]+x[2]] #region of the face
        cv2.rectangle(user_image, (x[0],x[1]), (x[0]+x[2],x[1]+x[3]), (255, 30, 30), 5) #draw rectagle around face
        roi = cv2.resize (roi, (100, 120)) #crop the image to the face
        roi_numpy= np.array(roi) #convert image to numpy array
        roi_numpy = roi_numpy.reshape((1,100,120,1)) #reshape the numpy array to have just 1 number for the color
        roi_numpy = roi_numpy/255 #reduce the integer for the color to make it easier for the computer to compute
        predictions=model.predict(roi_numpy) #use the model to make a prediction about the identitiy
        #predictions is a list with probability of being anushka and of being katrina
        final_predic=np.argmax(predictions[0]) #returns the index of the item in the list with the highest number
        if final_predic == 0: #if the index is 0...
            names.append("anushka") #...the person is anushka, add her name to the list
            cv2.putText(user_image, "anushka",(x[0],x[1]+x[3]+20), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 30, 30), 3) #add the name to the image
        elif final_predic == 1:
            names.append("katrina")
            cv2.putText(user_image, "katrina",(x[0],x[1]+x[3]+20), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 30, 30), 3)
    image_path = "static/"+random_name_generator() #create a path for the image to be in the static folder with a random name
    cv2.imwrite(image_path, user_image) #add the image to the static folder with a random name
    return names, image_path #return the names of the people in the image and the path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET": #if the user has asked to see the index route...
        return render_template ("index.html") #...show it
    
    else: #if the user has clicked the submit button...
        the_file_data = request.files["file"] #get the file
        file_String = the_file_data.read() #read content of file
        np_image = np.fromstring(file_String, np.uint8) #convert into numpy array from string
        image_data = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED) #decode it, read pixel by pixel, make it 2d array
        recog_names, image_path = recog_face(image_data) #predict
        return render_template("show_image.html", names = recog_names, image = image_path)#display the names and final photo on a new html page

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "GET": 
        return render_template ("admin.html")
    else:
        password = request.form["password"]
        if pbkdf2_sha256.verify(password, os.environ.get("password")):
            files = os.listdir("static")
            for fileName in files:
                os.remove("static/"+fileName)
            return "cool"
        else:
            return redirect ("/")
        


#this runs the app
if __name__ == "__main__":
    app.run()
