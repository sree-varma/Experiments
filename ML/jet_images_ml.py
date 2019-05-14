"""
The machine learning part is included in this code the deep CNN is created to classify the images
(https://arxiv.org/pdf/1612.01551.pdf) section 3.2.

"""
import os
import tensorflow as tf
import numpy as np
import argparse
import data

from network import inference,inference_1
import dataset

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-batch', '--batch-size-of-images', dest='batch_size', type=float, default=100, help='Batch size of images used for training')
parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=0.001, help='learning rate for training')
parser.add_argument('-drp', '--dropout-rate', dest='drp', type=float, default=0.9, help='dropout rate for training')
parser.add_argument('-num', '--number', dest='number', type=float, default=1, help='number of iterations required ')
parser.add_argument('-logpath', '--log-session-path', dest='logdir', type=str, default='/data/qcd/logs', help='Path to save the session')


parser.add_argument('-method', '--pre-method', dest='method', type=str, help='Preprocessing methods to be used! mit,deeptop_17,deeptop_18')
parser.add_argument('-signame', '--signame', dest='signame', type=str, help='Name of the background! tW,top,ZZ')
parser.add_argument('-bgname', '--background-name', dest='bgname', type=str,  help='Name of the background! jW,qcd,Zj')
parser.add_argument('-pt', '--ptrange', dest='ptrange', type=str, help='pt range! 350-400,500-550,1300-1400,500')
parser.add_argument('-R', '--radius', dest='R', type=str, help='Radius of the jet! 0.8,1.5')
parser.add_argument('-channels', '--number of color channels of the image', dest='channels', type=float, default=1, help='Number of color channels of the image')
parser.add_argument('-mass', '--whether mass is to be included into the network', dest='with_mass', type=float, default=1, help='Whether mass is to be added to the network')


parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,help='Print output on progress running script')

args = parser.parse_args()

verbose = args.verbose


method=args.method
signame=args.signame
bgname=args.bgname
ptrange=args.ptrange
R=args.R
channels=int(args.channels)
with_mass=int(args.with_mass)

print '\n'
print "TRAINING ON"

print signame,bgname,R,ptrange,"Colour-",channels," Mass-",with_mass,method,'\n'


 
path = '/data/qcd/data/'+signame+'_'+bgname+'/'+ptrange+'/'#plotting/'

trsig_image,mass_sig=data.load_data(path,data_name=signame,train_name='train',method=method,R=R,pt=ptrange)
print "train-signal done"
print trsig_image.shape,mass_sig.shape
trbg_image,mass_bg= data.load_data(path,data_name=bgname,train_name='train',method=method,R=R,pt=ptrange)
print "train-bg done"
print trbg_image.shape,mass_bg.shape

train_images=[trsig_image,trbg_image]
jet_mass= [mass_sig,mass_bg]

assert trsig_image.shape[0]==trbg_image.shape[0]
print "Image size: ", trsig_image.shape[0],trsig_image.shape[1]

r =0.00001


tf.reset_default_graph()

gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=1.0)

session =  tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))#tf.Session()#


args = parser.parse_args()

verbose = args.verbose


"""
Inputs
"""
convolutional = True

# image dimensions (only squares for now)
img_size = trsig_image.shape[0]

num_channels = channels
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = [signame,bgname ]
num_classes = len(classes)

if not convolutional:
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
else:
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0], img_shape[1], num_channels], name='x')



x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


tf.summary.image('input', x_image, 3)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

mass = tf.placeholder(tf.float32,shape=[None,1],name='mass')

"""
Inference (Forward Pass)
"""
drp= args.drp
if method =='mit':
	logits,features= inference(x_image, num_classes=num_classes,num_channels=num_channels,drp=[drp,drp,drp],mass=mass,with_mass=with_mass)

if method =='deeptop_18':
	logits,features= inference_1(x_image, num_classes=num_classes,num_channels=num_channels,drp=[drp,drp,drp],mass=mass,with_mass=with_mass)


print np.shape(mass)

y_probs =tf.nn.softmax(logits)
tf.summary.histogram('probs', y_probs)

y_pred = tf.argmax(logits, dimension=1)


"""
Learning configuration
"""

# batch size
batch_size = int(args.batch_size) # validation split
print "Batch size-", batch_size
validation_size = 0.2
learning_rate = args.learning_rate
print "learning rate-", learning_rate

# how long to wait after validation loss stops improving before terminating training
early_stopping =5 # use None if you don't want to implement early stopping


"""
Data set Configuration
"""

dataset=data.read_train_sets(train_images,img_size,num_channels,classes,mass=jet_mass,validation_size=validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(dataset.train.labels)))
print("- Validation-set:\t{}".format(len(dataset.valid.labels)))


num_batches = int(dataset.train.num_examples/batch_size)

"""
Evaluation Configuration
"""

valdiation_batches = 10
validation_freq = num_batches  # validate once per epoch for now (change to suit data)
train_summary_freq = num_batches 
saving_freq = num_batches  # save once per epoch for now (change to suit data)

"""
Loss Function + Optimization
"""

with tf.name_scope("cost"):
    cross_entropy = tf.keras.backend.categorical_crossentropy(output=logits,target=y_true,from_logits=True)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)
with tf.name_scope("Optimize"):
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = learning_rate
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000, 0.96, staircase=True)
	# Passing global_step to minimize() will increment it at each step.

	if method=='mit':
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	if method=='deeptop_18':
		optimizer =tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

session.run(tf.global_variables_initializer())

logdir = args.logdir    
print logdir
writer = tf.summary.FileWriter(logdir)
writer.add_graph(session.graph)

eval_writer = tf.summary.FileWriter(logdir + '_eval')  
x_batch, _,_  = dataset.train.next_batch(batch_size)

# Add image summary to inspect network input
tf.summary.image('input', x_batch)
merged_summary = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=150)

num = args.number
def optimize(num_iterations, starting_iteration=0):

    av_train_acc = 0.0

    for i in range(starting_iteration,
                   starting_iteration + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.

        x_batch, y_true_batch,mass_batch = dataset.train.next_batch(batch_size)


        if not convolutional:
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]
            x_batch = x_batch.reshape(batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.

        feed_dict_train = {x: x_batch, y_true: y_true_batch, mass:mass_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.

        # training step
        _, train_acc, train_loss = session.run([optimizer, accuracy, cost], feed_dict=feed_dict_train)
        av_train_acc = av_train_acc + train_acc

        if i % train_summary_freq == 0:
            summary_value = session.run(merged_summary, feed_dict_train)
            writer.add_summary(summary_value, i)
        if i % validation_freq == 0:
            av_validate_acc = 0.0
            av_val_loss = 0.0
            for j in range(valdiation_batches):
                x_valid_batch, y_valid_batch,valid_mass_batch = dataset.valid.next_batch(batch_size)
                if not convolutional:
                    x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)

                feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch,mass:valid_mass_batch}
                validate_acc, val_loss= session.run([accuracy, cost], feed_dict=feed_dict_validate)
                av_validate_acc = av_validate_acc + validate_acc
                av_val_loss = av_val_loss + val_loss
            av_val_loss /= valdiation_batches
            av_validate_acc /= valdiation_batches
            summary_value_val = session.run(merged_summary, feed_dict_validate)
            eval_writer.add_summary(summary_value_val, i)
        if i % saving_freq==0:
	    epoch = int(i / num_batches)
            if not os.path.exists(logdir+'/saved_models'):
                os.makedirs(logdir+'/saved_models')
            saver.save(session, logdir+'/saved_models/saved_model_{}.ckpt'.format(epoch))

        if i % num_batches == 0:
            epoch = int(i / num_batches)
            av_train_acc /= num_batches

            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}," \
                  " Training Loss: {3:.3f}, Validation Loss: {4:.3f}"
            print(msg.format(epoch + 1, av_train_acc, av_validate_acc, train_loss,  av_val_loss))
            av_train_acc = 0.0

    final_iteration = starting_iteration + num_iterations

    return final_iteration

optimize(num_iterations=int(num*(dataset.train.num_examples)))
