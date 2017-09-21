##Hasan Rafiq - GITHub
# Neural network classification on tensorflow
# Kaggle's forest cover prediction problem: To predict 7 types of forest cover as per input data
# 4 layer Deep Neural Net
# Use One hot encoder for output column hot encoding
# With Dropout setting
# With Dynamic learning rate
# Achieved Final Val Set accuracy of 84%
# Option to apply L2 Reg
# Option to apply batch norm
# Option to apply dropout
import numpy as np
import tensorflow as tf
import xlrd
import math

#Define parameters for the model
DATA_FILE = "train_set.xlsx"
TEST_FILE = "test_set.xlsx"
batch_size      = 30                #Size of temporary train set batched out from full train set
n_epochs        = 2000              #Number of full train set repetitions of trainings
epsilon         = 1e-3              #For Batch norm
lambda_loss     = 0.01              #Lambda for L2 reg
opt             = 1                 #0 for no norm or reg; 1 for L2; 2 for Batch norm

#Neural network structure
#L0 = 784
L1 = 100
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
learn_rate  = tf.placeholder(tf.float32,  name="learn_rate")
Y_un_he     = tf.placeholder(tf.int32,    name="un_encoded")  #Takes unencoded value
#Dropout: feed in 1 when testing, 0.75 when training
pkeep       = tf.placeholder(tf.float32, name="Dropout_pkeep")

#Tensorflow operation: Do one hot encoding on Y labels and cast to Float32
YonehotInt = tf.one_hot(Y_un_he, depth=7)
Y          = tf.cast(YonehotInt, tf.float32)

#Create tensorflow variables for weight and biases => Variables are trainable
#When using RELUs, make sure biases are initialised with small *positive* values for example 0.1
W1 = tf.get_variable("W1", [54,L1], initializer = tf.contrib.layers.xavier_initializer())
B1 = tf.Variable(tf.zeros([L1])/10000)                                   #Biases should be one in a ReLu NN
W2 = tf.get_variable("W2", [L1,L2], initializer = tf.contrib.layers.xavier_initializer())
B2 = tf.Variable(tf.zeros([L2])/10000)                                   #Biases should be one in a ReLu NN
W3 = tf.get_variable("W3", [L2,L3], initializer = tf.contrib.layers.xavier_initializer())
B3 = tf.Variable(tf.zeros([L3])/10000)                                   #Biases should be one in a ReLu NN
W4 = tf.get_variable("W4", [L3,L4], initializer = tf.contrib.layers.xavier_initializer())
B4 = tf.Variable(tf.zeros([L4])/10000)                                   #Biases should be one in a ReLu NN
W5 = tf.get_variable("W5", [L4,7], initializer = tf.contrib.layers.xavier_initializer())
B5 = tf.Variable(tf.zeros([7]))                                         #Seven outputs

#Create tensorflow computation code
if opt == 2:
####Start batch normalization code in layer1
# Layer 1 with BN, using Tensorflows built-in BN function X -> Y1_MAT -> Batch norm -> Y1_BN -> Y1
    Y1_MAT  = tf.matmul(X,   W1) + B1
    batch_mean, batch_var = tf.nn.moments(Y1_MAT,[0])
    scale   = tf.Variable(tf.ones([L1])) #Gamma - Multiplier
    beta    = tf.Variable(tf.zeros([L1])) #Beta - Offset
    Y1_BN   = tf.nn.batch_normalization(Y1_MAT,batch_mean,batch_var,beta,scale,epsilon)
    Y1      = tf.nn.relu(Y1_BN)
    Y1d     = Y1
####End batch normalization code in layer1
else:
    Y1  = tf.nn.relu(tf.matmul(X,   W1) + B1)
    Y1d = tf.nn.dropout(Y1, pkeep)                                  #Apply dropout probability in layer 2

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
entropy     =   tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y) #(input * 1 matrix )
loss_unreg  =   tf.reduce_mean(entropy)

#Apply L2 regularization and add to loss_unreg
if opt == 1:
    l2_loss = lambda_loss * (tf.nn.l2_loss(W1) + 
                                               tf.nn.l2_loss(W2) +
                                               tf.nn.l2_loss(W3) +
                                               tf.nn.l2_loss(W4) +
                                               tf.nn.l2_loss(W5))
    loss = loss_unreg + l2_loss
else:    
    loss = loss_unreg

#4. define training operation
# using adam optimizer with learning rate of 0.005 to minimize cost
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

#5. Only for logging -> accuracy of the trained model( Y Vs Y_ ), between 0 (worst) and 1 (best)
correct_prediction  = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

######Write graph to folder
def save_graph(tfs):
    export_path = 'out_graph'
    builder     = tf.saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(Y_un_he)

    #Prepare prediction signature
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'forest_data_in': tensor_info_x},
          outputs={'forest_cover_out': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))    

    #Add variables
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_forest':
              prediction_signature
      },
      legacy_init_op=legacy_init_op)

    #Save .PB file
    builder.save()

###### Start training iterations
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

        tot_train_acc = 0
        tot_train_loss = 0
        for b in range(n_batches):
            batch_end   = batch_start + batch_size
            X_batch     = Xdata[batch_start:batch_end,:]
            Y_batch     = Ydata[batch_start:batch_end]
            #print(X_batch.shape)
            #print(Y_batch.shape)

            # Dynamic LR => learning rate decay
            max_learning_rate   = 0.0005
            min_learning_rate   = 0.000001
            decay_speed         = 150.0 
            lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

            #Run hot encoding operation
            #Y_batch_enc = sess.run(Yonehot, feed_dict={Y_un_he: Y_batch})

            #Run training operation
            o_,a_,l_,cp = sess.run([optimizer,accuracy,loss,correct_prediction], feed_dict={X: X_batch, Y_un_he: Y_batch, pkeep: 1, learn_rate: lr })
            tot_train_acc = tot_train_acc + a_      #complete train set accuracy
            tot_train_loss = tot_train_loss + l_    #complete train set loss
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
        #Yt_batch_enc = sess.run(Yonehot, feed_dict={Y_un_he: Yt_data})        
        acc_,lt_     = sess.run([accuracy,loss], feed_dict={X: Xt_data, Y_un_he: Yt_data, pkeep: 1}) 
        print("Last learning rate: ", str(lr))
        print("Epoch " + str(i) + " Training loss: " , str(tot_train_loss / n_batches))
        print("Epoch " + str(i) + " Test loss: " , str(lt_))
        print("Epoch " + str(i) + " Train accuracy: " + str(tot_train_acc * 100/n_batches) + " %")
        print("Epoch " + str(i) + " Validation accuracy:  " + str(acc_*100) + " %")

    ############################################
    save_graph(sess)
