<b>Count number of people in an image using Google's Tensorflow Object Detection model</b>

Help Link: https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb

Below models are available which you can replace in the code to test and predict.

<b>Images to predict:</b>
Add new images with naming convention "Image<X>.jpg" in the test_images folder

<b>Model name:	<br/></b>
ssd_mobilenet_v1_coco	fast	=> Fastest but accuracy is +<br/>
ssd_inception_v2_coco	fast	=> Slower but accuracy is ++<br/>
rfcn_resnet101_coco	medium	=> Slower but accuracy is +++<br/>
faster_rcnn_resnet101_coco	=> Slower but accuracy is ++++<br/>
faster_rcnn_inception_resnet_v2_atrous_coco => Slowest but accuracy is +++++<br/>
