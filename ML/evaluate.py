import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve ,auc
import matplotlib.pyplot as plt
import pandas as pd
from network import inference
import dataset


tf.reset_default_graph()
session = tf.Session()
path='/data/ML/jet_images/plots/'
path1='/data/ML/jet_images/features/'

"""
Inputs
"""
convolutional = True

# image dimensions (only squares for now)
img_size = 33

num_channels = 3
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = ['quarks', 'gluons']
num_classes = len(classes)

if not convolutional:
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
else:
    x = tf.placeholder(tf.float32, shape=[None, img_shape[0], img_shape[1], num_channels], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
tf.summary.image('input', x_image, 3)
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


"""
Inference (Forward Pass)
"""

logits, features = inference(x_image, num_classes=num_classes,num_channels=num_channels)

y_probs = tf.nn.softmax(logits)
tf.summary.histogram('probs', y_probs)

y_pred = tf.argmax(logits, dimension=1)


"""
Restore variables
"""

saver = tf.train.Saver()
saver.restore(session,  tf.train.latest_checkpoint('/data/ML/jet_images/Herwig/200-220/colour/saved_models/'))#event_generators/ Herwig/grayscale/200-220/saved_models/'))

"""
Data set Configuration
"""
test_path='/usr/Pythia/colour/images_200-220GeV/test/'#'/usr/images_generators/train/'
print test_path
test_images, test_ids, labels = dataset.read_test_set(test_path, img_size,num_channels, classes)

#test_path1 = '/usr/Herwig/colour/images_200-220GeV/test/'#/usr/images_generators/test/'#'data/jet_images_sample/test'
#test_images1,test_ids1,labels1=dataset.read_test_set(test_path1,img_size,classes)

batch_size =128

print("Size of:")
print("- Test-set:\t\t{}".format(len(test_images)))

num_batches = int(len(test_images)/batch_size)

predictions = []
true_labels = []
#predictions1 = []
#true_labels1= []
features_out=[]
for i in range(num_batches):
    test_image_batch = test_images[i * batch_size:(i+1) * batch_size]
    #test_image_batch1 = test_images1[i * batch_size:(i+1) * batch_size]

    test_label_batch = labels[i * batch_size:(i+1) * batch_size]
    #test_label_batch1 = labels1[i * batch_size:(i+1) * batch_size]

    feed_dict = {x: test_image_batch, y_true:  test_label_batch}
    #feed_dict1 = {x: test_image_batch1, y_true:  test_label_batch1}

    y_pred_value, features_value = session.run([y_probs, features], feed_dict=feed_dict)
    
    #print features_value
    #y_pred_value1, features_value1 = session.run([y_probs, features], feed_dict=feed_dict1)


    #TODO: save features for reuse later
    predictions.extend(y_pred_value)
    true_labels.extend(test_label_batch)#list(np.argmax(test_label_batch, axis=1)))
    features_out.extend(features_value)
    
    #predictions1.extend(y_pred_value1)
    #true_labels1.extend(test_label_batch1)#list(np.argmax(test_label_batch, axis=1)))

features_out=np.asarray(features_out)
print np.shape(features_out)
#np.savez(path+"features_test_herwig_colour_200",features_out) 

true_labels=np.asarray(true_labels)
predictions=np.asarray(predictions)
print np.shape(true_labels)
print np.shape(predictions)


#np.savez(path+"labels_test_herwig_colour_200.npz",true_labels)

#true_labels1=np.asarray(true_labels1)
#predictions1=np.asarray(predictions1)

#print("Accuracy: {}".format(accuracy_score(true_labels, predictions)))
#print("Precision_score: {}".format(precision_score(true_labels, predictions)))
#print("Recall_score: {}".format(recall_score(true_labels, predictions)))
#print("F1_score: {}".format(f1_score(true_labels, predictions)))
#fpr, tpr, thresholds = roc_curve(true_labels, predictions)

fprs, tprs = [None] * 2, [None] * 2

aucs = [None] * 2

#fprs1, tprs1 = [None] * 2, [None] * 2
#aucs1 = [None] * 2

for i in range(2):
    	fprs[i], tprs[i], _ = roc_curve((true_labels[:, i]),predictions[:,i])
#	fprs1[i],tprs1[i],_=roc_curve((true_labels1[:, i]),predictions1[:,i])
	aucs[i] = auc(fprs[i], tprs[i], reorder=True)
#	aucs1[i] = auc(fprs1[i], tprs1[i], reorder=True)
print fprs[0]

aucs = auc(fprs[0],tprs[0],reorder=True)

#aucs1 = auc(fprs1[0],tprs1[0],reorder=True)

fprs_quarks_herwig_df = pd.DataFrame({"fprs_quarks_herwig":fprs[0]})
tprs_quarks_herwig_df= pd.DataFrame({"tprs_quarks_herwig":tprs[0]})


#fprs_quarks_pythia_df = pd.DataFrame({"fprs_quarks_pythia":fprs1[0]})
#tprs_quarks_pythia_df= pd.DataFrame({"tprs_quarks_pythia":tprs1[0]})



fprs_quarks_herwig_df.to_csv(path1+"Herwig_fprs_pyth_200_color.csv",index=False)
tprs_quarks_herwig_df.to_csv(path1+"Herwig_tprs_pyth_200_color.csv",index=False)
	


print aucs

#plt.plot([0, 1], [0, 1], '--', color='black')
#plt.xlabel('Quark Jet Efficiency(True Positive)')
#plt.ylabel('Gluon Jet Rejection(False Negative)')


#plt.plot(tprs[0],1-fprs[0], label='%s (AUC %.2lf)' % ("Herwig CNN on Herwig color images", aucs))

#plt.plot(tprs1[0],1-fprs1[0], label='%s (AUC %.2lf)' % ("Pythia CNN on Pythia color images", aucs1))
#plt.legend(fontsize=10,loc=3)
#plt.show()
#plt.figure()
#plt.plot(tpr,1-fpr)
#plt.show()

signals = []
backgrounds = []
for i in xrange(len(true_labels[:,0])):
  y = true_labels[i,0]
  if y ==1:
    signals.append(predictions[i,0])
  if y ==0:
    backgrounds.append(predictions[i,0])



bins = np.linspace(-0.5, 1.5, 100)

plt.hist(signals, bins, alpha=0.4, label='signal')
plt.hist(backgrounds, bins, alpha=0.4, label='background')
plt.legend(loc='upper right')
plt.savefig(path+"np_Hp_200c.png",dpi=300)
plt.show()
