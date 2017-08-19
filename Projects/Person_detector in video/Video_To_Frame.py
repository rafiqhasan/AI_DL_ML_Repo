#Shivam Srivastava
#Hasan Rafiq
import cv2
import datetime
vidcap = cv2.VideoCapture('Video.mp4')
success,image = vidcap.read()
count = 0
success = True

 # Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#print ((cv2.__version__).split('.'))
 
if int(major_ver)  < 3 :
    fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Version " + major_ver )
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print ("Version " + major_ver )
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
 
intfps = int(fps)
#intfps = intfps/2 
FrameCounter=0

now_time = datetime.datetime.now()
now_timeaddition=now_time
IsleID=200
while success:
  success,image = vidcap.read()
  if count%intfps==0:
  	 print ('Read new frame number',count,': ', success)
  	 
  	 cv2.imwrite("C:\\Users\\hrafiq\\Documents\\HRafiq_Deloitte\\FIRM_Initiatives\\DarkWeb_CCTV\\Images\\Frame%d.jpg" %(FrameCounter), image) # save frame as JPEG file
  	 FrameCounter+=1
  	 #now_timeaddition+=datetime.timedelta(minutes = 10)
  	 #print (str(now_timeaddition))
  
  count += 1

print (" Successfully Captured " ,FrameCounter ,"Frames")

vidcap.release();

print("Calling Tensor.py")

import Tensor


