#Hasan Rafiq

#Using - Google's object detection Tensorflow model to count number of people in an image

import numpy as np

import fnmatch
import cv2
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



from collections import defaultdict

from io import StringIO

from matplotlib import pyplot as plt

from PIL import Image



# What model to download.

MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

MODEL_FILE = MODEL_NAME + '.tar.gz'

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'



# Path to frozen detection graph. This is the actual model that is used for the object detection.

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'



# List of the strings that is used to add correct label for each box.

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90



#Download Model

if 1 == 2:

    opener = urllib.request.URLopener()

    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    tar_file = tarfile.open(MODEL_FILE)

    for file in tar_file.getmembers():

        file_name = os.path.basename(file.name)

        if 'frozen_inference_graph.pb' in file_name:

            tar_file.extract(file, os.getcwd())


dirpath="C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Images"
bodypath="C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Bodies\\"


#Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()

with detection_graph.as_default():

    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

        serialized_graph = fid.read()

        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name='')

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
        print("Image size")
        print(im_width)
        print(im_height)

        print("Box coords")
        print(left)
        print(right)
        print(top)
        print(bottom)

        #Draw line on image
        draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill='red')

        img_crop = image.crop((left, top, right, bottom))
        #img_crop.show()
        #img_crop.save( bodypath + "_cropped_" + str(fname) + ".jpg")     

def load_image_into_numpy_array(image):

  (im_width, im_height) = image.size

  return np.array(image.getdata()).reshape(

      (im_height, im_width, 3)).astype(np.uint8)



# For the sake of simplicity we will use images:

# image1.jpg

# image2.jpg

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
Number_of_images= (len(fnmatch.filter(os.listdir(dirpath), '*.jpg')))

print("Number of images found in folder is " + str(Number_of_images))

PATH_TO_TEST_IMAGES_DIR = 'Images'

TEST_IMAGE_PATHS = [ os.path.join(dirpath, 'frame{}.jpg'.format(i)) for i in range(0, Number_of_images) ]



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
    
   # conn.request("POST", "/Retail_Dark/storePresenceGetData.xsjs", payload, headers)
    #res = conn.getresponse()
    #data = res.read()
    #print(data.decode("utf-8"))


    for image_path in TEST_IMAGE_PATHS:

      image_idx = image_idx + 1
      print("... Analyzing frame: " + str(image_idx-1))

      #image_path = "C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Images\\Frame0.jpg"
      image = Image.open(image_path)

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
          if ( classes[0,r] == 1 ) & ( scores[0,r] > 0.6 ):
              persons = persons + 1
              #Hasan -> Extract body from image
              print("Boxes")
              box_xy = np.reshape(boxes[0,r], [1, 4])
              #print(box_xy.shape)
              #print(box_xy)
              image_new = get_body_from_image(image_new,box_xy,r)

      #Write marked image to directory
      image_new.save(bodypath + str(image_idx-1) + "_body_marked.jpg")

      #print(scores.shape)

      print("People in image" + str(image_idx) + ": " + str(persons))
      print("Time written:" ,now_plus_10.time())
      frame_out = "Health2017-09-31T" + str(now_plus_10.time().strftime('%H%M%S'))
      payload1={}
      payload = "{\n\t\"STOREID\":\"250\",\n\t\"DATE\":\"09/31/2017\",\n\t\"TIME\":\""+str(now_plus_10.time())+"\",\n\t\"COUNT\":\""+str(persons)+"\",\n\t\"LOCATION\":\"INDIA\",\n\t\"CAMERAID\":\"200\",\n\t\"AISLEID\":\"30\",\n\t\"CAMERAID\":\"30\",\n\t\"FRAMEID\":\""+str(FrameIDcount)+"\"}"
      headers = {'authorization': "Basic dmlzaGd1cHRhOlBhc3N3b3JkLjE=", 'cache-control': "no-cache",}

      #Save frames in a folder
      print("Frame save" + str(frame_out))
      print("payload" + str(payload))
      #image.save("C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Img\\" + str(frame_out) + ".jpeg") # save frame as JPEG file
      #conn = http.client.HTTPConnection("ussltcsnl1283.solutions.glbsnet.com:8000")
      #conn.request("POST", "/ZIOT_ANA_RET/ZIOT_RET_XS_PROJ/DARK_ANALYTICS/storePresenceGetData.xsjs", payload, headers)
      #res = conn.getresponse()
      #data = res.read()
      #print(data.decode("utf-8"))
      now_plus_10 = now_plus_10 + datetime.timedelta(minutes = 10) #interval
      FrameIDcount+=1

      '''
			Payload:
			{
			              "STOREID":"100",
			              "DATE":"04/12/2017",
			              "TIME":now_plus_10.time(),
			              "COUNT":persons,
			              "LOCATION":"INDIA",
			              "CAMERAID":"100",
			              "AISLEID":"10"
			}
			'''


