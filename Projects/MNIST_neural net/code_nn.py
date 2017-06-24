#Hasan -> Logistic regression on tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Define parameters for the model
learning_rate = 0.005
batch_size = 100
n_epochs = 5

#Neural network structure
#L0 = 784
L1 = 500
L2 = 200
L3 = 100
L4 = 50

#Read input data
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

#Prepare placeholders
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

#Create tensorflow variables for weight and biases => Variables are trainable
W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))        #(1,50)
B1 = tf.Variable(tf.ones([L1])/10)                                  #Biases should be one in a ReLu NN
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))         #(50,40)
B2 = tf.Variable(tf.ones([L2])/10)                                  #Biases should be one in a ReLu NN
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))         #(40,40)
B3 = tf.Variable(tf.ones([L3])/10)                                  #Biases should be one in a ReLu NN
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))         #(40,20)
B4 = tf.Variable(tf.ones([L4])/10)                                  #Biases should be one in a ReLu NN
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))          #(20,10)
B5 = tf.Variable(tf.zeros([10]))                                    #Ten outputs

#Create tensorflow computation code
Y1  = tf.nn.relu(tf.matmul(X,   W1) + B1)
Y2  = tf.nn.relu(tf.matmul(Y1,  W2) + B2)
Y3  = tf.nn.relu(tf.matmul(Y2,  W3) + B3)
Y4  = tf.nn.relu(tf.matmul(Y3,  W4) + B4)

#1. Predict Y
# the model that returns probability distribution of possible label of the image
# through the softmax layer
# a batch_size x 10 tensor that represents the possibility of the digits
Ylogits = tf.matmul(Y4,  W5) + B5       #(inputsize * 10 matrix)

#2. Convert logit to predicted probabilities = Y_
Y_ = tf.nn.softmax(Ylogits)             #(inputsize * 10 matrix)

#3. use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
entropy =   tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y) #(input * 1 matrix )
loss    =   tf.reduce_mean(entropy)

#4. define training operation
# using adam optimizer with learning rate of 0.01 to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#5. Only for logging -> accuracy of the trained model( Y Vs Y_ ), between 0 (worst) and 1 (best)
correct_prediction  = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training iterations
with tf.Session() as sess:
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    #sess.run(init)
    sess.run(tf.global_variables_initializer())
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs): # train the model n_epochs times
        for _ in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            o_,l_,a_,yl_,y__,cp_,e_ = sess.run([optimizer,loss,accuracy,Ylogits,Y_,correct_prediction,entropy], feed_dict={X: X_batch, Y:Y_batch})
            print("Accuracy on train set in epoch: " + str(i) + " batch: " + str(_) + " is: " + str(a_*100) + "%")

            #Run prediction on test set => Run session.run on last training run so that weights and biases are retained
            X_test, Y_test = MNIST.test.next_batch(batch_size)
            acc_ = sess.run([accuracy], feed_dict={X: X_test, Y:Y_test})
            print("Accuracy on test set in epoch: " + str(i) + " batch: " + str(_) + " is: " + str(acc_[0]*100) + "%")
