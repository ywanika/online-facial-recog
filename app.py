from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
import numpy as np
import cv2
import os
import random
app = Flask(__name__)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model ("famousFace.h5")


def recog_face(user_image):
    image = cv2.imread(user_image)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
    for x in faces:
        roi = im_gray[x[1] : x[1]+x[3], x[0] : x[0]+x[2]]
        roi = cv2.resize (roi, (100, 120))
        roi_numpy= np.array(roi)
        roi_numpy = roi_numpy.reshape((1,100,120,1))
        roi_numpy = roi_numpy/255
        predictions=model.predict(roi_numpy)
        final_predic=np.argmax(predictions[0])
        if final_predic == 0:
            cv2.putText(image, "anushka",(x[0],x[1]+x[3]+20), cv2.FONT_HERSHEY_DUPLEX, .8, (100,100,255), 1) 
        elif final_predic == 1:
            cv2.putText(image, "katrina",(x[0],x[1]+x[3]+20), cv2.FONT_HERSHEY_DUPLEX, .8, (100,100,255), 1)
    cv2.imwrite("static/recoged_image.png", image)
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template ("index.html")
    
    else:
        the_file_data = request.files["file"]
        the_file_enc = secure_filename(the_file_data.filename)
        the_file_data.save(the_file_enc)
        image = recog_face(the_file_enc)
        return redirect("/show_image")

@app.route ("/show_image")
def show_image():
    return render_template ("show_image.html")


#this runs the app
if __name__ == "__main__":
    app.run(debug=True)
