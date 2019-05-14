import matplotlib
#matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve,auc
import matplotlib.pyplot as plt
import argparse
from network import inference
import dataset
import data
import scikitplot as skplt
import pandas as pd
import pickle
from scipy import interpolate
from network_1 import inference_1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-method', '--pre-method', dest='method', type=str, help='Preprocessing methods to be used! mit,deeptop_17,deeptop_18')
parser.add_argument('-signame', '--signame', dest='signame', type=str, help='Name of the background! tW,top,ZZ')
parser.add_argument('-bgname', '--background-name', dest='bgname', type=str,  help='Name of the background! jW,qcd,Zj')
parser.add_argument('-pt', '--ptrange', dest='ptrange', type=str, help='pt range! 350-400,500-550,1300-1400,500')
parser.add_argument('-R', '--radius', dest='R', type=str, help='Radius of the jet! 0.8,1.5')
parser.add_argument('-channels', '--number of color channels of the image', dest='channels', type=float, default=1, help='Number of color channels of the image')
parser.add_argument('-logpath', '--log-session-path', dest='logdir', type=str, default='/data/qcd/logs', help='Path to save the session')
parser.add_argument('-mass', '--whether mass is to be included into the network', dest='with_mass', type=float, help='Whether mass is to be added to the network')

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
print "TESTING ON"
print signame,bgname,R,ptrange,"Colour-",channels," Mass-",with_mass,method,'\n'




tf.reset_default_graph()
session = tf.Session()

path = '/data/qcd/data/'+signame+'_'+bgname+'/'+ptrange+'/'#plotting/'

tesig_image,mass_sig=data.load_data(path,data_name=signame,train_name='train',method=method,R=R,pt=ptrange)
print "train-signal done"
print tesig_image.shape
tebg_image,mass_bg = data.load_data(path,data_name=bgname,train_name='train',method=method,R=R,pt=ptrange)
print "train-bg done"
print tebg_image.shape

test_images=[tesig_image,tebg_image]
jet_mass=[mass_sig,mass_bg]
print mass_sig

assert tesig_image.shape[0]==tebg_image.shape[0]
print "Image size: ", tesig_image.shape[0],tesig_image.shape[1]




logdir = args.logdir    
print logdir
"""
Inputs
"""
convolutional = True

# image dimensions (only squares for now)
img_size = tesig_image.shape[0]
num_channels = 1
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

charge =tf.placeholder(tf.float32,shape=[None,1],name='mass')


"""
Inference (Forward Pass)
"""
keep_prob1 = tf.placeholder("float")
if method =='mit':
	logits,features= inference(x_image, num_classes=num_classes,num_channels=num_channels,drp=[1.0,1.0,1.0],mass=mass,with_mass=with_mass)

if method =='deeptop_18':
	logits,features= inference_1(x_image, num_classes=num_classes,num_channels=num_channels,drp=[1.0,1.0,1.0],mass=mass,with_mass=with_mass)

y_probs = tf.nn.softmax(logits)
tf.summary.histogram('probs', y_probs)
y_pred = tf.argmax(logits, dimension=1)

percentage=1.0


"""
Restore variables
"""

saver = tf.train.Saver()
saver.restore(session,  tf.train.latest_checkpoint( logdir+'/saved_models'))

"""
Data set Configuration
"""

test_images, labels,jet_mass = data.read_test_set(test_images,img_size,num_channels,classes,jet_mass)
batch_size = 100
print("Size of:")
print("- Test-set:\t\t{}".format(len(test_images)))

num_batches = int(len(test_images)/batch_size)

predictions = []
true_labels = []
mass_jet = []
charge_jet = []
for i in range(num_batches):
    test_image_batch = test_images[i * batch_size:(i+1) * batch_size]
    test_label_batch = labels[i * batch_size:(i+1) * batch_size]
    mass_batch = jet_mass[i*batch_size:(i+1)*batch_size]
    

    feed_dict = {x: test_image_batch, y_true:  test_label_batch,mass: mass_batch}
    y_pred_value,feature= session.run([y_probs,features], feed_dict=feed_dict)

    #TODO: save features for reuse later
    predictions.extend(y_pred_value)
    true_labels.extend(test_label_batch)
    mass_jet.extend(mass_batch)
    

true_labels=np.asarray(true_labels)
predictions=np.asarray(predictions)
a,b= np.shape(predictions)

sig_pred = predictions[:,0]
sig_pred = sig_pred.reshape((a,1))
mass_jet = np.asarray(mass_jet)
print np.shape(mass_jet)


"""
print("Accuracy: {}".format(accuracy_score(true_labels, predictions)))
print("Precision_score: {}".format(precision_score(true_labels, predictions)))
print("Recall_score: {}".format(recall_score(true_labels, predictions)))
print("F1_score: {}".format(f1_score(true_labels, predictions)))
"""


true_n = 0.
total_n = 0.
for true, pred in zip(true_labels,predictions):
    total_n = total_n + 1
    if pred[0] > pred[1] and true[0] > true[1]:
        true_n = true_n + 1
    if pred[1] > pred[0] and true[1] > true[0]:
        true_n = true_n + 1

print "Accuracy: ", float(true_n/total_n)

signals = []
backgrounds = []
truth=[]
for i in xrange(len(true_labels[:,0])):
  y = true_labels[i,0]
  if y ==1:
    signals.append(predictions[i,0])
    truth.append(1.)
  if y ==0:
    backgrounds.append(predictions[i,0])
    truth.append(0)


fprs, tprs, thresholds = roc_curve(true_labels[:,0], predictions[:,0])

aucs = auc(fprs,tprs)
fprs = fprs+0.00000001
tprs = tprs
fprs = np.divide(1,fprs)
f = interpolate.interp1d(tprs, fprs)
print "Background rejection at signal efficiency of 0.3: ", f(0.3)

fprs, tprs, thresholds = roc_curve(true_labels[:,0], predictions[:,0])


aucs = auc(fprs,tprs)
fprs = fprs+0.00000001
tprs = tprs
fprs = np.divide(1,fprs)

print "AUC: ",aucs

"""

plt.plot([0, 1], [0, 1], '--', color='black')
plt.yscale('log')

plt.ylim([0.1, 8*10**3])
plt.xlim([-0.05, 1.05])
plt.xlabel('Top quark Tagging Efficiency')
plt.ylabel('QCD Rejection Factor')
plt.plot(tprs,fprs, color='darkorange',lw=2, label='ROC curve (area = %0.2f)'  %aucs)
plt.grid(True)


plt.legend(fontsize=10,loc=3)
plt.show()




bins = np.linspace(-0.5, 1.5, 100)

plt.hist(signals, bins, alpha=0.4, label='signal')
plt.hist(backgrounds, bins, alpha=0.4, label='background')
plt.legend(loc='upper right')
plt.savefig("results_1.png",dpi=300)

plt.show()
"""
"""
with open('/data/qcd/master/TopTagging/TopTagging/PlottingROCwithFPRTPR/results_350/fpr_350-400_mit1', 'wb') as fp:
  pickle.dump(fprs, fp)
with open('/data/qcd/master/TopTagging/TopTagging/PlottingROCwithFPRTPR/results_350/tpr_350-400_mit1', 'wb') as fp:
  pickle.dump(tprs, fp)

"""
