import tensorflow as tf
import argparse
import sys 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
mnist_var = None

def conv_relu(feature, kernels, biases):
	
	#convolution + relu operation over a given input
	conv = tf.nn.conv2d(feature, kernels, strides=[1, 1, 1, 1], padding="SAME")
	return tf.nn.relu(conv + biases)

def max_pooling(value):

	#max pooling operation
	return tf.nn.max_pool(value=value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(feature, weights, biases, act=tf.nn.relu):

	#activated data
	return act(tf.matmul(feature, weights) + biases)

def inference(x):
	global mnist_var

	#image reshaped 
	x_img = tf.reshape(x, [-1, 28, 28, 1])

	with tf.variable_scope("conv_relu_1"):
		conv_relu1 = conv_relu(x_img, mnist_var[0], mnist_var[1])

	maxp1 = max_pooling(conv_relu1)

	with tf.variable_scope("conv_relu_2"):
		conv_relu2 = conv_relu(maxp1, mnist_var[2], mnist_var[3])
	
	maxp2 = max_pooling(conv_relu2)
	maxp2 = tf.reshape(maxp2, [-1, 7 * 7 * 64])

	with tf.variable_scope("fc_layer_1"):
		fc_1 = fc_layer(maxp2, mnist_var[4], mnist_var[5])
	
	with tf.variable_scope("fc_layer_2"):
		fc_2 = fc_layer(fc_1, mnist_var[6], mnist_var[7], act=tf.identity)

	return fc_2

def main(_):
	global mnist_var

	#MNIST datasets
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	#input placeholder
	with tf.name_scope("input"):
		x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

	with tf.Session( ) as session:
		
		#restoring meta-graph from training mnist model
		with tf.name_scope("saver"):
			saver = tf.train.import_meta_graph(FLAGS.ckpt_dir + "/train/mnist_model.meta")
		
		#restoring session
		saver.restore(session, FLAGS.ckpt_dir + "/train/mnist_model")

		#validation variables
		with tf.variable_scope("mnist_validation"):
		
			#variables of training process
			with tf.name_scope("trained_variables"):
				mnist_var = tf.trainable_variables( )

			#model architecture
			with tf.name_scope("model"):
				conv_inf =  inference(x)

			with tf.name_scope("read_out"):
				soft = tf.nn.softmax(conv_inf)
			
			img = Image.open(FLAGS.image).convert("L")
			
			img_conv = np.asarray(img) / 256.
			img_conv = img_conv.reshape([1, 784])

			res = session.run(soft, feed_dict={x: img_conv})

			print("input image is %g" % np.argmax(res))

			plt.imshow(img, cmap="gray")
			plt.show( )
			
			# #model accuracy
			# with tf.name_scope("accuracy"):
			# 	correct_prediction = tf.equal(tf.argmax(conv_inf, 1), tf.argmax(y_, 1))
			# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			# #number of batches
			# num_bathces = int(mnist.test.num_examples / FLAGS.mini_batch_size)
			# acc = []
			
			# print("printing test accuracy...")

			# for i in range(num_bathces):
				
			# 	#test mini-batch
			# 	batch = mnist.test.next_batch(50)		
				
			# 	#calculating accuracy
			# 	res = session.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
				
			# 	acc.append(res)

			# #calculatiing and printing total accuracy 
			# total_acc = np.mean(acc) * 100.
			# print("Total test accuracy: %g" % total_acc)
		
if __name__ == '__main__':

	parser = argparse.ArgumentParser( )
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,
						help='If true, uses fake data for unit testing.')
	parser.add_argument('--mini_batch_size', type=int, default=50, 
						help='Mini batch size.')
	parser.add_argument('--image', type=str, default="seven.png", 
						help='Test image.')
	parser.add_argument('--ckpt_dir', type=str, default='checkpoint',
						help='Checkpoint directory.')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='Summaries log directory.')
	parser.add_argument('--data_dir', type=str, default='MNIST_data',
                      help='Directory for storing input data')

	FLAGS, unparsed = parser.parse_known_args( )
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)