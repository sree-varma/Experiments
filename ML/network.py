import tensorflow as tf
import numpy as np
"""
Helper Functions
"""
keep_prob1 = tf.placeholder("float")

def conv_layer(input, num_input_channels, filter_size, num_filters,drp,name="conv"):
    with tf.name_scope(name):
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        initializer = tf.keras.initializers.he_uniform(seed=None)
        w = tf.Variable(initializer(shape=shape), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        print np.shape(conv)
	conv = tf.nn.dropout(conv,drp)
        act = tf.nn.relu(conv + b)
	act = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act


def fc_layer(input, num_inputs, num_outputs, name="fc"):
    with tf.name_scope(name):
        initializer = tf.keras.initializers.he_uniform(seed=None)
        w = tf.Variable(initializer(shape=[num_inputs, num_outputs]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_outputs]), name="B")
        act = (tf.matmul((input), w) +b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
    return act

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


"""
Network Definition
"""


def inference(image, num_classes,num_channels,drp,mass,with_mass):
    # Convolutional Layer 1.filters
    filter_size1 = 8
    num_filters1 = 64
    # Convolutional Layer 2.
    filter_size2 = 4
    num_filters2 = 64

    # Convolutional Layer 3.
    filter_size3 = 4
    num_filters3 =64

    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.

    # Number of color channels for the images: 1 channel for gray-scale.
    num_channels = num_channels

    conv1 = conv_layer(input=image, num_input_channels=num_channels,
                       filter_size=filter_size1, num_filters=num_filters1,drp=drp[0],name="conv1")
    conv2 = conv_layer(input=conv1, num_input_channels=num_filters1,
                       filter_size=filter_size2, num_filters=num_filters2,drp=drp[1],name="conv2")
    conv3 = conv_layer(input=conv2, num_input_channels=num_filters2,
                       filter_size=filter_size3, num_filters=num_filters3,drp=drp[2],name="conv3")
    layer_flat, num_features = flatten_layer(conv3)
    layer=tf.concat([layer_flat,mass],axis=1)

    if with_mass==1:
    	fc1 = fc_layer(input=layer, num_inputs=num_features+1, num_outputs=fc_size, name="fc1")
	print 'WITH MASS'
    else:
    	fc1 = fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, name="fc1")
	print 'WITHOUT MASS'

    activated_fc1 = tf.nn.relu(fc1)

  
    logits = fc_layer(input=activated_fc1, num_inputs=fc_size, num_outputs=num_classes, name="fc2")

    return logits,fc1





def block(input,filters,filter_size1,filter_size2,num_filters1,num_filters2,drp,name="block",conv_1="conv1",conv_2="conv2"):	

	conv1 = conv_layer(input=input,num_input_channels=filters,filter_size=filter_size1,num_filters=num_filters1,drp=1.0,name=conv_1)
	conv2 = conv_layer(input=conv1, num_input_channels=num_filters1,filter_size=filter_size2, num_filters=num_filters2,drp=1.0,name=conv_2)
	return conv2


def inference_1(image, num_classes,num_channels,drp,mass,with_mass=True):

 	num_channels = num_channels
	# Fully-connected layer.
	fc_size1 = 64  # Number of neurons in fully-connected layer.
	fc_size2 = 256  # Number of neurons in fully-connected layer.
	fc_size3 = 256  # Number of neurons in fully-connected layer.
	#Number of color channels for the images: 1 channel for gray-scale.
	num_filters= 64


	filter_size1 = 4
        num_filters1 = 128

        # Convolutional Layer 2.
	filter_size2 = 4
	num_filters2 = 64

	filter_size3 = 4
        num_filters3 = 64

        # Convolutional Layer 2.
	filter_size4 = 4
	num_filters4 = 64
	
	block1 = block(image,filters=num_channels,filter_size1=filter_size1,filter_size2=filter_size2,num_filters1=num_filters1,num_filters2=num_filters2,drp=drp[0],name="Block1",conv_1="conv1",conv_2="conv2")

	pooling = tf.layers.max_pooling2d(block1, pool_size=(2, 2), strides=(2, 2), padding="SAME")

	block2 = block(pooling,filters=num_filters,filter_size1=filter_size3,filter_size2=filter_size4,num_filters1=num_filters3,num_filters2=num_filters4,drp=drp[1],name="Block2",conv_1="conv3",conv_2="conv4")
	a,b,c,d = np.shape(block2)

	pooling1 = tf.layers.max_pooling2d(block2, pool_size=(2, 2), strides=(2, 2), padding="SAME")
	layer_flat, num_features = flatten_layer(pooling1)

	fc1 = fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size1, name="fc1")

	activated_fc1 = tf.nn.relu(fc1)
	
	fc2 = fc_layer(input=activated_fc1, num_inputs=fc_size1, num_outputs=fc_size2, name="fc2")
	
	activated_fc2 = tf.nn.relu(fc2)
        if with_mass==1:
		activated_fc2 = tf.concat([activated_fc2,mass],axis=1)
		fc3 = fc_layer(input=activated_fc2, num_inputs=fc_size2+1, num_outputs=fc_size3, name="fc3")
		print 'WITH MASS'
	else:	
		fc3 = fc_layer(input=activated_fc2, num_inputs=fc_size2, num_outputs=fc_size3, name="fc3")
		print 'WITHOUT MASS'
	activated_fc3 = tf.nn.relu(fc3)

	logits = fc_layer(input=activated_fc3, num_inputs=fc_size3, num_outputs=num_classes, name="output")

	return logits, fc1










