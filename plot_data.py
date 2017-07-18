import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

with tf.Session( ) as session:
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
	
	saver = tf.train.import_meta_graph("savings/mnist_model.ckpt.meta")
	saver.restore(session, "savings/mnist_model.ckpt")

	img = mnist.train.images[10]
	img_reshaped = img.reshape([1, 28, 28, 1])

	mnist_var = tf.trainable_variables( )
	conv = tf.nn.conv2d(x, mnist_var[0], strides=[1, 1, 1, 1], padding="SAME")
	conv_res = session.run(conv, feed_dict={x: img_reshaped})

	for i in range(64):
		pic = conv_res[0][:, :, i]
		
		plt.subplot(8, 8, i + 1)
		plt.imshow(pic, cmap="gray")
		
		frame1 = plt.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)

	plt.show( )