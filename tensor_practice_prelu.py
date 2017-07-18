import argparse
import sys 
import tensorflow as tf
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def PReLU(feature, shape=[1]):
	#trainable parameter from PReLU
	a = tf.get_variable("a", shape=shape, initializer=tf.constant_initializer(0.25))

	#return PReLU activation
	return tf.nn.relu(feature) + tf.mul(tf.minimum(tf.constant(0.), feature), a)

def conv_relu(features, weight_shape, bias_shape):

	#create weights
	weights = tf.get_variable("weights", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))	

	#create biases
	biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.1))

	#creates convolution operation
	conv = tf.nn.conv2d(features, weights, strides=[1, 1, 1, 1], padding="SAME")

	#activated data
	return PReLU(conv + biases, bias_shape)

def max_pooling(value):

	#max pooling operation over a given value
	return tf.nn.max_pool(value=value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(features, weight_shape, bias_shape, act=PReLU):

	#creates weights
	weights = tf.get_variable("weights", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))	

	#create biases
	biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.1))

	#activated data
	return act(tf.matmul(features, weights) + biases)

def inference(x, keep_prob):
	#reshape input image batch and creates a tensor
	x_img = tf.reshape(x, [-1, 28, 28, 1])

	with tf.variable_scope("conv_relu_1"):
		conv_relu1 = conv_relu(x_img, [5, 5, 1, 32], [32])

	with tf.name_scope("max_pool_1"):
		maxp1 = max_pooling(conv_relu1)

	with tf.variable_scope("conv_relu2"):
		conv_relu2 = conv_relu(maxp1, [5, 5, 32, 64], [64])

	with tf.name_scope("max_pool_2"):
		maxp2 = max_pooling(conv_relu2)

	#flattens feature to 1-dimension
	maxp2 = tf.reshape(maxp2, [-1, 7 * 7 * 64])

	with tf.variable_scope("fc_layer1"):
		fc_1 = fc_layer(maxp2, [7 * 7 * 64, 1024], [1024])
		
		#dropout operation to minimize overfitting
		drop_fc_1 = tf.nn.dropout(fc_1, keep_prob)

	with tf.variable_scope("fc_layer2"):
		fc_2 = fc_layer(drop_fc_1, [1024, 10], [10], act=tf.identity)

	#returns readout layer
	return fc_2

def main(_):
	#extract datasets
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	#input placeholders for training and accuracy operations
	with tf.name_scope("input"):
		x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="batch")
		y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")
	
	#input placeholder for keep probability in dropout operation 
	with tf.name_scope("keep_probability"):
		keep_prob = tf.placeholder(dtype=tf.float32, name="dropout")

	#main variables and operations
	with tf.variable_scope("mnist") as scope:
		
		#model architecture 
		with tf.name_scope("model"):
			y_conv = inference(x, keep_prob)

		#loss function for training process
		with tf.name_scope("loss"):
			diff = tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)
			loss = tf.reduce_mean(diff)

		#training step operation
		with tf.name_scope("training"):
			train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

		#calculates model accuracy over a given batch
		with tf.name_scope("accuracy"):
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#create saver object to restore model variables
		saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="mnist"))

		#main session
		with tf.Session( ) as session:

			#initialize all variables
			tf.global_variables_initializer( ).run( )

			#training process over pre-defined epochs
			for j in range(FLAGS.epochs):
				print("epoch %d initiated" %(j + 1))

				#calculates how many mini-batches are present in training dataset	
				num_batches = int(mnist.train.num_examples / FLAGS.mini_batch_size)

				#train whole batch using mini-batches of pre-defined size
				for i in range(num_batches):

					#takes a batch from training dataset
					batch = mnist.train.next_batch(FLAGS.mini_batch_size)

					#run training operation given a mini-batch of images
					session.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout})

				#calculates how many mini-batches are present in validation dataset	
				num_batches = int(mnist.validation.num_examples / FLAGS.mini_batch_size)

				#train whole batch using mini-batches of pre-defined size
				for i in range(num_batches):

					#takes a batch from training dataset
					batch = mnist.validation.next_batch(FLAGS.mini_batch_size)

					#run training operation given a mini-batch of images
					session.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.dropout})

				#save model context at every epoch
				save_path = saver.save(session, FLAGS.ckpt_dir + '/prelu_model')
				print("epoch %s saved in file %s" %((j + 1), save_path))

				print("calculating test accuracy...")

				#calculating test accuracy data
				#number of batches
				num_bathces = int(mnist.test.num_examples / FLAGS.mini_batch_size)

				acc = []
				
				for i in range(num_bathces):
					
					#test mini-batch
					batch = mnist.test.next_batch(50)		
					
					#calculating accuracy
					res = session.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) * 100.

					acc.append(res)

				#calculatiing and printing total accuracy 
				total_acc = np.mean(acc)
				print("Total test accuracy: %g" % total_acc)

if __name__ == '__main__':

	parser = argparse.ArgumentParser( )
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,
						help='If true, uses fake data for unit testing.')
	parser.add_argument('--epochs', type=int, default=20,
						help='Number of training epochs.')
	parser.add_argument('--mini_batch_size', type=int, default=50, 
						help='Mini batch size.')
	parser.add_argument('--learning_rate', type=float, default=1e-4,
						help='Initial learning rate value.')
	parser.add_argument('--dropout', type=float, default=0.5,
						help='Keep probability for dropout operation.')
	parser.add_argument('--beta', type=float, default=5e-5,
						help='Beta constant that regards to L2 regularization.')
	parser.add_argument('--ckpt_dir', type=str, default='checkpoint/train/PReLU',
						help='Checkpoint directory.')
	parser.add_argument('--log_dir', type=str, default='logs',
						help='Summaries log directory.')
	parser.add_argument('--data_dir', type=str, default='MNIST_data',
                      help='Directory for storing input data')

	FLAGS, unparsed = parser.parse_known_args( )
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)