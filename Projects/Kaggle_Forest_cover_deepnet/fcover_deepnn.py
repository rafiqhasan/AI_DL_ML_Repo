##Hasan Rafiq - GITHub
# Neural network classification on tensorflow
# Kaggle's forest cover prediction problem: To predict 7 types of forest cover as per input data
# 4 layer Deep Neural Net
# Use One hot encoder for output column hot encoding
# With Dropout setting
# With Dynamic learning rate
# Achieved Final Test Set accuracy of 86% => Can be improvised with more data
import numpy as np
import tensorflow as tf
import xlrd
import math

#Define parameters for the model
DATA_FILE = "train_set.xlsx"
TEST_FILE = "test_set.xlsx"
batch_size      = 30                #Size of temporary train set batched out from full train set
n_epochs        = 2000              #Number of full train set repetitions of trainings

#Neural network structure
#L0 = 784
L1 = 400
L2 = 300
L3 = 300
L4 = 200

#Read train data file
print("Loading data files ... ")
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
file_data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#Read test data file
book = xlrd.open_workbook(TEST_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
test_data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
#test_data = file_data[0:200,:]
#t_samples = sheet.nrows - 1

#Prepare placeholders
#For X, Y, learning rate and dropout
X           = tf.placeholder(tf.float32,  name="data")
Y           = tf.placeholder(tf.float32,  name="label_hotencoded")
learn_rate  = tf.placeholder(tf.float32,  name="learn_rate")
Y_un_he     = tf.placeholder(tf.int32,    name="un_encoded")  #Takes unencoded value
#Dropout: feed in 1 when testing, 0.75 when training
pkeep       = tf.placeholder(tf.float32, name="Dropout_pkeep")

#Tensorflow operation: Do one hot encoding on Y labels and cast to Float32
YonehotInt = tf.one_hot(Y_un_he, depth=7)
Yonehot    = tf.cast(YonehotInt, tf.float32)

#Create tensorflow variables for weight and biases => Variables are trainable
#When using RELUs, make sure biases are initialised with small *positive* values for example 0.1
W1 = tf.Variable(tf.random_uniform([54, L1], minval=-1, maxval=1))         #(54,L1)
B1 = tf.Variable(tf.ones([L1])/10000)                                  #Biases should be one in a ReLu NN
W2 = tf.Variable(tf.random_uniform([L1, L2], minval=-1, maxval=1))         #
B2 = tf.Variable(tf.ones([L2])/10000)                                  #Biases should be one in a ReLu NN
W3 = tf.Variable(tf.random_uniform([L2, L3], minval=-1, maxval=1))         #
B3 = tf.Variable(tf.ones([L3])/10000)                                  #Biases should be one in a ReLu NN
W4 = tf.Variable(tf.random_uniform([L3, L4], minval=-1, maxval=1))         #
B4 = tf.Variable(tf.ones([L4])/10000)                                  #Biases should be one in a ReLu NN
W5 = tf.Variable(tf.truncated_normal([L4, 7], stddev=0.1))          #(L4,7)
B5 = tf.Variable(tf.zeros([7]))                                     #Seven outputs

#Create tensorflow computation code
Y1  = tf.nn.relu(tf.matmul(X,   W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)                                      #Apply dropout probability in layer 1
Y2  = tf.nn.relu(tf.matmul(Y1d,  W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)                                      #Apply dropout probability in layer 2
Y3  = tf.nn.relu(tf.matmul(Y2d,  W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)                                      #Apply dropout probability in layer 3
Y4  = tf.nn.relu(tf.matmul(Y3d,  W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)                                      #Dropout not applied in this layer

#1. Predict Y
# the model that returns probability distribution of possible label of the image
# through the softmax layer
# a batch_size x 10 tensor that represents the possibility of the digits
Ylogits = tf.matmul(Y4d,  W5) + B5       #(inputsize * 10 matrix)

#2. Convert logit to predicted probabilities = Y_
Y_ = tf.nn.softmax(Ylogits)              #(inputsize * 10 matrix)

#3. use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
entropy =   tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y) #(input * 1 matrix )
loss    =   tf.reduce_mean(entropy)

#4. define training operation
# using adam optimizer with learning rate of 0.005 to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

#5. Only for logging -> accuracy of the trained model( Y Vs Y_ ), between 0 (worst) and 1 (best)
correct_prediction  = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training iterations
with tf.Session() as sess:
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(n_samples/batch_size)
    print("Starting neural training.....")
    for i in range(n_epochs):   #Train the model n_epochs times
        #Batch Restart
        batch_start = 0

        #RANDOM Shuffling on data from train file to Xdata and Ydata with 
        rand_arr = np.arange(n_samples)     #Generate array of integers
        np.random.shuffle(rand_arr)         #Random sort array
        rand_data = file_data[rand_arr]     #Copy data from random array
        Xdata = rand_data[:,0:54]           #[X,54] => Read column 1 to 54 in Excel
        Xdata = np.mat(Xdata)               #[X,54] => Conversion to matrix
        Ydata = rand_data[:,54]             #[Y,1]  => Output column as array

        for b in range(n_batches):
            batch_end   = batch_start + batch_size
            X_batch     = Xdata[batch_start:batch_end,:]
            Y_batch     = Ydata[batch_start:batch_end]
            #print(X_batch.shape)
            #print(Y_batch.shape)

            # Dynamic LR => learning rate decay
            max_learning_rate   = 0.0001
            min_learning_rate   = 0.000001
            decay_speed         = 400.0 
            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

            #Run hot encoding operation
            Y_batch_enc = sess.run(Yonehot, feed_dict={Y_un_he: Y_batch})

            #Run training operation
            o_,a_,l_,cp = sess.run([optimizer,accuracy,loss,correct_prediction], feed_dict={X: X_batch, Y: Y_batch_enc, pkeep: 1, learn_rate: lr })
            #print(l_)
            #print("Accuracy on train set in epoch: " + str(i) + " batch: " + str(b) + " is: " + str(a_*100) + "% loss: " + str(l_))

            #Pick next batch
            batch_start = batch_start + batch_size

        ############################################
        #Run prediction on test set after full epoch
        #=> Run session.run on last training run so that weights and biases are retained                                       
        Xt_data = test_data[:,0:54]           #[X,54] => Read column 1 to 54 in Excel
        Xt_data = np.mat(Xt_data)             #[X,54] => Conversion to matrix
        Yt_data = test_data[:,54]             #[Y,1]  => Output column as array

        #Run operations=> Hot encoding + prediction accuracy
        Yt_batch_enc = sess.run(Yonehot, feed_dict={Y_un_he: Yt_data})        
        acc_,lt_     = sess.run([accuracy,loss], feed_dict={X: Xt_data, Y: Yt_batch_enc, pkeep: 1})
        print("Last learning rate: ", str(lr))
        print("Last Training loss: " , str(l_))
        print("Current Test loss: " , str(lt_))
        print("Accuracy on last train set in epoch: " + str(i) + " was " + str(a_*100) + "% loss: " + str(l_))
        print("Accuracy on full test  set in epoch: " + str(i) + " is: " + str(acc_*100) + " %")
