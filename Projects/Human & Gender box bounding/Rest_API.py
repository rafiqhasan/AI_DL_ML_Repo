#Hasan Rafiq

#Using - Object detection Tensorflow model to count number of people in an image and custom trained gender detection models

import numpy as np
#import fnmatch
import os
import six.moves.urllib as urllib
import sys
import tarfile
import PIL.ImageDraw as ImageDraw
import tensorflow as tf
import zipfile
import http.client
import datetime

#import Image
from werkzeug import secure_filename
from collections import defaultdict
from io import StringIO
from io import BytesIO
#from matplotlib import pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from PIL import Image

#Parameters
app = Flask(__name__)
serialized_graph = ''
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Function to load existing Gender Tflow graph
modelFullPath = 'gender/model_gender.pb'
labelsFullPath = 'gender/output_labels.txt'
gender_graph = tf.Graph()
detection_graph = tf.Graph()

def load_existing_gender_graph():
#Load Tflow custom gender model    
    with gender_graph.as_default():
        od_graph_def2 = tf.GraphDef()
        with tf.gfile.GFile(modelFullPath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def2.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def2, name='')

def load_existing_human_graph():
#Load Tflow custom gender model    
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')          

# Run existing graph on new image
def classify_gender(image):
    #Read image file into binary
    #image_data = tf.gfile.FastGFile(imagePath, 'rb').read()
    image_array = np.array(image)[:, :, 0:3]  # Select RGB channels only

    #Run TF Session -> classifier
    with tf.Session(graph=gender_graph) as sess2:
        softmax_tensor = sess2.graph.get_tensor_by_name('final_result:0')
        predictions = sess2.run(softmax_tensor,
                               {'DecodeJpeg:0': image_array})
        predictions = np.squeeze(predictions)

    #Convert prediction probability vector to label texts ( Trained on )
    f = open(labelsFullPath, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    for x in range(0,predictions.shape[0]):
        human_string = labels[x]
        score = predictions[x]
        print('%s (score = %.5f)' % (human_string, score))
    return(np.argmax(predictions),predictions[np.argmax(predictions)])            

def initalize_server():
#Download Model
    if 1 == 2:
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

#Load tensorflow models
    load_existing_human_graph()
    load_existing_gender_graph()        

#Function to crop out body predictions from images
def get_body_from_image(image,box_coord,fname):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    #Get number of boxes and process each box
    for i in range(box_coord.shape[0]):
        #Get X / Y min and max
        ymin = box_coord[i,0]
        xmin = box_coord[i,1]
        ymax = box_coord[i,2]
        xmax = box_coord[i,3]
    
        #Generate coordinates from normalized coo-ords
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)

        #crop image
        if (1 == 2):
            print("Image size")
            print(im_width)
            print(im_height)

            print("Box coords")
            print(left)
            print(right)
            print(top)
            print(bottom)

        #Gender classification
        print("Classifying gender ... ")
        img_crop = image.crop((left, top, right, bottom))
        gender ,score = classify_gender(img_crop)
        if (gender == 1 and score > 0.6): #male
            #Draw line on image
            print("male")
            draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill='red')
        elif (gender == 0 and score > 0.6): #female
            #Draw line on image
            print("female")
            draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill='green')
        else:
            #Draw line on image
            print("Unclear")
            draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill='yellow')  
        
    return image
    #img_crop.show()
    #img_crop.save( bodypath + "_cropped_" + str(fname) + ".jpg")     

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

@app.route('/body_api', methods=['POST'])
def input():
    print("Request received ... ")
    f = request.files['file']
    filename = secure_filename(f.filename)
    # For the sake of simplicity we will use images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    now = datetime.datetime.now()
    now_plus_10=now
    now_plus_10 = now_plus_10 #+ datetime.timedelta(minutes = 120)
    FrameIDcount=0

    #Detector main code
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_idx = 0
        #conn = http.client.HTTPConnection("ussltcsnl1283.solutions.glbsnet.com:8000")
        
        #conn.request("POST", "/Retail_Dark/storePresenceGetData.xsjs", payload, headers)
        #res = conn.getresponse()
        #data = res.read()
        #print(data.decode("utf-8"))
          
        print("... Analyzing frame: ")
        #image_path = "C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Images\\Frame0.jpg"
        image = Image.open(f.stream)
        print(image.size)
       
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
       
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
       
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
       
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
       
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        persons = 0
        print(boxes.shape)
       
        for r in range(0,classes.shape[1]):
            image_new = image
            #Check if class = 1(Person)
            if ( classes[0,r] == 1 ) & ( scores[0,r] > 0.6 ): 
                persons = persons + 1
                #Hasan -> Extract body from image
                box_xy = np.reshape(boxes[0,r], [1, 4])
                image_new = get_body_from_image(image_new,box_xy,r)
       
        #Send analyzed image as response
        print("People in image : " + str(persons))
        img_io = BytesIO()
        image_new.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    initalize_server()
    app.run(host='localhost', port=5000)
