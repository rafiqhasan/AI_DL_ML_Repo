#Hasan -> Neural net implementation
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import math

DATA_FILE = "slr05.xls"
lr = 0.0008
iter = 10000

#Neural network structure
L1 = 50
L2 = 40
L3 = 40
L4 = 20

#Read data file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#1. Copy data from file to X and Y
X = data[:,0]       #[42,]
X = np.mat(X).T     #[42,1] => Conversion to matrix
Y = data[:,1]       #[42,]
Y = np.mat(Y).T     #[42,1] => Conversion to matrix

#2. Create tensorflow X and Y placeholders
XTF = tf.placeholder(tf.float32,name="InputX")
YTF = tf.placeholder(tf.float32,name="Actual_out")

#3. Create tensorflow variables for weight and biases => Variables are trainable
W1 = tf.Variable(tf.truncated_normal([1, L1], stddev=0.1))          #(1,50)
B1 = tf.Variable(tf.ones([L1])/10)                                  #Biases should be one in a ReLu NN
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))         #(50,40)
B2 = tf.Variable(tf.ones([L2])/10)                                  #Biases should be one in a ReLu NN
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))         #(40,40)
B3 = tf.Variable(tf.ones([L3])/10)                                  #Biases should be one in a ReLu NN
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))         #(40,20)
B4 = tf.Variable(tf.ones([L4])/10)                                  #Biases should be one in a ReLu NN
W5 = tf.Variable(tf.truncated_normal([L4, 1], stddev=0.1))          #(20,1)
B5 = tf.Variable(tf.zeros([1]))                                     #One output

#4. Create tensorflow computation code
Y1  = tf.nn.relu(tf.matmul(XTF, W1) + B1)
Y2  = tf.nn.relu(tf.matmul(Y1,  W2) + B2)
Y3  = tf.nn.relu(tf.matmul(Y2,  W3) + B3)
Y4  = tf.nn.relu(tf.matmul(Y3,  W4) + B4)
Y_  = tf.nn.relu(tf.matmul(Y4,  W5) + B5)

#5. Calcuate lost as MSE ( Actual - prediction ) power 2
cost = tf.square(YTF - Y_)

#6. Optimizer function => Minimize final cost 
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#7. Run tensor graph
with tf.Session() as sess:
    #Invoke tensorboard logging
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())

    #8. Train model, start iteration
    for i in range(1, iter):
        _opt, _cost, _y = sess.run([opt,cost,Y_], feed_dict={XTF: X, YTF: Y})
        err = math.sqrt(np.sum(_cost) / n_samples) #Error after Ith iteration
        print("Cost at iteration " + str(i) + " is " + str(err))

    #9. output the values of W and B
    #w_value, b_value = sess.run([W3, B3])
    #print(w_value)
    #print(b_value)
    print(_y)
    sess.close()
