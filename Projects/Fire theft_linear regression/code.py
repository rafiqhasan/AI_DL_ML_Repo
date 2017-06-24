import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


DATA_FILE = "slr05.xls"
lr = 0.001
iter = 500

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
XTF = tf.placeholder(tf.float32,name="Input")
YTF = tf.placeholder(tf.float32,name="Actual_out")

#3. Create tensorflow variables for weight and biases => Variables are trainable
W = tf.Variable(tf.zeros([1,1], tf.float32),name="Weight")
B = tf.Variable(tf.zeros([1], tf.float32),name="Bias")

#4. Create tensorflow computation code
Y_ = tf.matmul(XTF, W) + B

#5. Calcuate lost as MSE
cost = tf.square(YTF - Y_)

#6. Optimizer function => Minimize final cost 
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

#7. Run tensor graph
with tf.Session() as sess:
    #Invoke tensorboard logging
    writer = tf.summary.FileWriter('./graphs', sess.graph)                                   
    sess.run(tf.global_variables_initializer())

    #Train model, start iteration
    for i in range(1, iter):
        sess.run(opt, feed_dict={XTF: X, YTF: Y})

    # Step 9: output the values of W and B
    w_value, b_value = sess.run([W, B])
    print(w_value)
    print(b_value)
    #print(np.sum(cost.eval()) / n_samples)  #Average cost
    #print(X*w_value + b_value) #Predicted values
