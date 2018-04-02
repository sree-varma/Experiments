import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
#import pp
import os
import sys
from multiprocessing import Process , Queue
import pixel_c
import cv2
import scipy.misc
import matplotlib.image


start_time = time.time()  # Time to execute the program.
path=''# Path of the program

#PATH OF FILES
#--------------
#1.Quarks
#--------

pathq1='/usr/Herwig/data/'
pathq2='/usr/Herwig/data/' 
pathq3='/usr/Herwig/data/'


#2.Gluons
#---------
pathg1 = '/usr/Herwig/data/'
pathg2='/usr/Herwig/data/' 

#3.Images
#--------

train_qimage_path='/usr/Herwig/colour/images_200-220GeV/train/quarks/'#'/home/k1629656/Jets/herwig_100/train/quarks'#'/home/k1629656/Jets/train/quarks/'#'/usr/Herwig/gluon_jets/images_100-110GeV'#\
test_qimage_path='/usr/Herwig/colour/images_200-220GeV/test/quarks/'#'/home/k1629656/Jets/herwig_100/test/quarks'#'/home/k1629656/Jets/test/quarks/'


train_gimage_path='/usr/Herwig/colour/images_200-220GeV/train/gluons/'
test_gimage_path='/usr/Herwig/colour/images_200-220GeV/test/gluons/'

#FILES
#======
#1.QUARKS
#--------

quarkFile1='herwig_gg2qqbar_40000_200-220_jet_image.dat'  # The generated data from Herwig simulation
quarkFile2='herwig_qq2qq_40000_200-220_jet_image.dat'
quarkFile3='herwig_qqbar2qqbar_40000_200-220_jet_image.dat'
df_quark1 = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
df_quark2 = pd.read_csv(pathq2+quarkFile2,dtype=None,delimiter=",")
df_quark3 = pd.read_csv(pathq3+quarkFile3,dtype=None,delimiter=",")
df_quark = pd.concat([df_quark1,df_quark2,df_quark3],ignore_index=True,axis=0)

gquark= df_quark.loc[df_quark['ETA']=='@@@'] 
gi_quark = gquark.index # Indices of '@@@'#
#gi_quark=gi_quark[:120]


#2.GLUONS
#---------

gluonFile1='herwig_gg2gg_60000_200-220_jet_image.dat'  # The generated data from Herwig simulation
gluonFile2 ='herwig_qqbar2gg_60000_200-220_jet_image.dat'
df_gluon1 = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
df_gluon2 = pd.read_csv(pathg2+gluonFile2,dtype=None,delimiter=",")

df_gluon = pd.concat([df_gluon1,df_gluon2],ignore_index=True,axis=0)
#df_gluon.to_csv('herwig_gloun_jet_120000_100-110_jet_image.dat',index = False)

ggluon= df_gluon.loc[df_gluon['ETA']=='@@@'] 
gi_gluon = ggluon.index # Indices of '@@@'
#gi_gluon=gi_gluon[:120]

gquark1= df_quark1.loc[df_quark1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark1 = gquark1.index # Indices of '@@@'##
#gi_quark1= gi_quark1[:40]
gquark2= df_quark2.loc[df_quark2['ETA']=='@@@'] 
gi_quark2 = gquark2.index
#gi_quark2=gi_quark2[:40] 
gquark3= df_quark3.loc[df_quark3['ETA']=='@@@'] 
gi_quark3 = gquark3.index 
#gi_quark3=gi_quark3[:40]

ggluon1= df_gluon1.loc[df_gluon1['ETA']=='@@@']
gi_gluon1 = ggluon1.index
#gi_gluon1=gi_gluon1[:60] 
ggluon2= df_gluon2.loc[df_gluon2['ETA']=='@@@'] 
gi_gluon2 = ggluon2.index 
#gi_gluon2=gi_gluon2[:60]


delta = 0.8/33 # length of each pixel
boxmax=0.4
r = 0.00001
total_pT = 0.00


quark_new=np.zeros((33,33))
gluon_new = np.zeros((33,33))


a1=np.size(gi_quark1)
a2=np.size(gi_quark2)
a3=np.size(gi_quark3)
b1=np.size(gi_gluon1)
b2=np.size(gi_gluon2)

quark1_c1= np.zeros((33,33,a1),dtype=float)
quark1_c2= np.zeros((33,33,a1),dtype=float)
quark1_c3= np.zeros((33,33,a1),dtype=float)

quark2_c1= np.zeros((33,33,a2),dtype=float)
quark2_c2= np.zeros((33,33,a2),dtype=float)
quark2_c3= np.zeros((33,33,a2),dtype=float)

quark3_c1= np.zeros((33,33,a3),dtype=float)
quark3_c2= np.zeros((33,33,a3),dtype=float)
quark3_c3= np.zeros((33,33,a3),dtype=float)


gluon1_c1= np.zeros((33,33,b1),dtype=float)
gluon1_c2= np.zeros((33,33,b1),dtype=float)
gluon1_c3= np.zeros((33,33,b1),dtype=float)

gluon2_c1= np.zeros((33,33,b2),dtype=float)
gluon2_c2= np.zeros((33,33,b2),dtype=float)
gluon2_c3= np.zeros((33,33,b2),dtype=float)


gluon_image= np.zeros((33,33,3,(b1+b2)),dtype=float)
x_g,y_g,z_g,n_g=np.shape(gluon_image)

print ("Read quark and gluon files")


if __name__=='__main__':
    q1 = Queue()
    p1=Process(target=pixel_c.image_array,args=(gi_quark1,quark1_c1,quark1_c2,quark1_c3,df_quark1,q1))
    p1.start()
    q2 = Queue()
    p2=Process(target=pixel_c.image_array,args=(gi_quark2,quark2_c1,quark2_c2,quark2_c3,df_quark2,q2))
    p2.start()
    q3 = Queue()
    p3=Process(target=pixel_c.image_array,args=(gi_quark3,quark3_c1,quark3_c2,quark3_c3,df_quark3,q3))
    p3.start()         
    g1= Queue()
    p4=Process(target=pixel_c.image_array,args=(gi_gluon1,gluon1_c1,gluon1_c2,gluon1_c3,df_gluon1,g1))
    p4.start()
    g2= Queue()
    p5=Process(target=pixel_c.image_array,args=(gi_gluon2,gluon2_c1,gluon2_c2,gluon2_c3,df_gluon2,g2))
    p5.start()
   

quark1_c1,quark1_c2,quark1_c3=q1.get()

quark1_c1=np.array(quark1_c1).reshape((33,33,a1))
quark1_c2=np.array(quark1_c2).reshape((33,33,a1))
quark1_c3=np.array(quark1_c3).reshape((33,33,a1))


quark2_c1,quark2_c2,quark2_c3=q2.get()

quark2_c1=np.array(quark2_c1).reshape((33,33,a2))
quark2_c2=np.array(quark2_c2).reshape((33,33,a2))
quark2_c3=np.array(quark2_c3).reshape((33,33,a2))


quark3_c1,quark3_c2,quark3_c3=q3.get()

quark3_c1=np.array(quark3_c1).reshape((33,33,a3))
quark3_c2=np.array(quark3_c2).reshape((33,33,a3))
quark3_c3=np.array(quark3_c3).reshape((33,33,a3))

print "Quark image array created!"

#gluon1_c1,gluon1_c2,gluon1_c3=pixel_c.image_array(gi_gluon1,gluon1_c1,gluon1_c2,gluon1_c3,df_gluon1)
#gluon2_c1,gluon2_c2,gluon2_c3=pixel_c.image_array(gi_gluon2,gluon2_c1,gluon2_c2,gluon2_c3,df_gluon2)


gluon1_c1,gluon1_c2,gluon1_c3=g1.get()
gluon2_c1,gluon2_c2,gluon2_c3=g2.get()

gluon1_c1=np.array(gluon1_c1).reshape((33,33,b1))
gluon1_c2=np.array(gluon1_c2).reshape((33,33,b1))
gluon1_c3=np.array(gluon1_c3).reshape((33,33,b1))

gluon2_c1=np.array(gluon2_c1).reshape((33,33,b2))
gluon2_c2=np.array(gluon2_c2).reshape((33,33,b2))
gluon2_c3=np.array(gluon2_c3).reshape((33,33,b2))

print "Gluon image array created!"
quark1 = np.concatenate((quark1_c1,quark1_c2,quark1_c3),axis=0)
quark2 = np.concatenate((quark2_c1,quark2_c2,quark2_c3),axis=0)
quark3 = np.concatenate((quark3_c1,quark3_c2,quark3_c3),axis=0)

gluon1 = np.concatenate((gluon1_c1,gluon1_c2,gluon1_c3),axis=0)
gluon2 = np.concatenate((gluon2_c1,gluon2_c2,gluon2_c3),axis=0)


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()


#QUARKS#



quark01=np.zeros((99,33,(a1+a2+a3)),dtype=float)
quark01[:,:,:a1]=quark1
quark01[:,:,a1:a1+a2]=quark2
quark01[:,:,a1+a2:a1+a2+a3]=quark3

count=0
quark= np.zeros((99,33,(a1+a2+a3)),dtype=float)
print quark.shape
for i in range(0,(a1+a2+a3)):
	if np.sum(quark01[:,:,i])!=0:
		quark[:,:,count] = quark01[:,:,i]
		count=count+1

print count
quark = quark[:,:,:count]
quark =pixel_c.shuffle(quark)
print quark.shape

quark_c1=quark[:33,:,:]
quark_c2=quark[33:66,:,:]
quark_c3=quark[66:,:,:]


gluon01=np.zeros((99,33,(b1+b2)),dtype=float)
gluon01[:,:,:b1]=gluon1
gluon01[:,:,b1:(b1+b2)]=gluon2

count1=0
gluon= np.zeros((99,33,(b1+b2)),dtype=float)
print gluon.shape
for i in range(0,(b1+b2)):
	if np.sum(gluon01[:,:,i])!=0:
		gluon[:,:,count1] = gluon01[:,:,i]
		count1=count1+1

print count1
gluon = gluon[:,:,:count1]

gluon=pixel_c.shuffle(gluon)
print gluon.shape

gluon_c1=gluon[:33,:,:]
gluon_c2=gluon[33:66,:,:]
gluon_c3=gluon[66:,:,:]


gluon_image= np.zeros((33,33,3,(count1)),dtype=float)
x_g,y_g,z_g,n_g=np.shape(gluon_image)

for index in range (0,count1):
	gluon_image[:,:,:,index]= np.dstack((gluon_c1[:,:,index],gluon_c2[:,:,index],gluon_c3[:,:,index]))

print gluon_image.shape

quark_image = np.zeros((33,33,3,(count)),dtype=float)
x_q,y_q,z_q,n_q=np.shape(quark_image)

for index in range(0,count):
	quark_image[:,:,:,index]= np.dstack((quark_c1[:,:,index],quark_c2[:,:,index],quark_c3[:,:,index]))
print quark_image.shape



images=np.zeros((99,33,(200000)),dtype=float)
images[:,:,:100000]=quark[:,:,:100000]
images[:,:,100000:]=gluon[:,:,:100000]
images_c1=images[:33,:,:]
images_c2=images[33:66,:,:]
images_c3=images[66:,:,:]

mean_c1=np.mean(images_c1,axis=2)
mean_c2=np.mean(images_c2,axis=2)
mean_c3=np.mean(images_c3,axis=2)

mean_image = np.dstack((mean_c1,mean_c2,mean_c3))
print mean_image.shape

std_c1=np.std(images_c1,axis=2)
std_c2=np.std(images_c2,axis=2)
std_c3=np.std(images_c3,axis=2)

std_image = np.dstack((std_c1,std_c2,std_c3))
images=0
images_c1=images_c2=images_c3=0
mean_c1=mean_c2=mean_c3=std_c1=std_c2=std_c3=0



#Plotting average image before zero centering and standardizing
#==============================================================#
mean_quark_c1= np.mean(quark_c1[:,:,:100000],axis=2)
mean_quark_c2= np.mean(quark_c2[:,:,:100000],axis=2)
mean_quark_c3= np.mean(quark_c3[:,:,:100000],axis=2)
mean_quark= np.dstack((mean_quark_c1,mean_quark_c2,mean_quark_c3))#Average of all images

std_quark_c1= np.std(quark_c1[:,:,:100000],axis=2)
std_quark_c2= np.std(quark_c2[:,:,:100000],axis=2)
std_quark_c3= np.std(quark_c3[:,:,:100000],axis=2)
std_quark= np.dstack((mean_quark_c1,mean_quark_c2,mean_quark_c3))#Std of all images


mean_gluon_c1=np.mean(gluon_c1[:,:,:100000],axis=2)
mean_gluon_c2=np.mean(gluon_c2[:,:,:100000],axis=2)
mean_gluon_c3=np.mean(gluon_c3[:,:,:100000],axis=2)
mean_gluon=np.dstack((mean_gluon_c1,mean_gluon_c2,mean_gluon_c3))

std_gluon_c1=np.std(gluon_c1[:,:,:100000],axis=2)
std_gluon_c2=np.std(gluon_c2[:,:,:100000],axis=2)
std_gluon_c3=np.std(gluon_c3[:,:,:100000],axis=2)
std_gluon=np.dstack((std_gluon_c1,std_gluon_c2,std_gluon_c3))

mean=np.zeros((33,33,3,2),dtype=float)
mean[:,:,:,0]=mean_quark
mean[:,:,:,1]=mean_gluon
#mean_image = np.mean(mean,axis=3)
#mean_image = np.mean(mean_image,axis=2)
#print mean_image.shape
std =np.zeros((33,33,3,2),dtype=float)
std[:,:,:,0]=std_quark
std[:,:,:,1]=std_gluon
#std_image=np.std(std,axis=3)
#std_image=np.std(std_image,axis=2)
print "Mean images creating"

cmap =plt.get_cmap('jet')


av1_imqfile ='av_image_before_step4-5_herwig3_4_5_color_.jpeg'
scipy.misc.imsave(os.path.join(path,av1_imqfile),mean_quark)

av1_imgfile ='av_image_before_step4-5_herwig1_2_color_.jpeg'
scipy.misc.imsave(os.path.join(path,av1_imgfile), mean_gluon)

print "Creating jet images :) "




for index in range(0,count1): # For all values and their corresponding indices in g_i
	gluon_image[:,:,:,index]=gluon_image[:,:,:,index]-mean_image
	gluon_image[:,:,:,index]=gluon_image[:,:,:,index]/(std_image+r)

	
av_gluon=np.mean(gluon_image,axis=3)

av2_imgfile ='av_image_herwig1_2_color.jpeg'
scipy.misc.imsave(os.path.join(path,av2_imgfile),av_gluon)


for index in range(0,count): # For all values and their corresponding indices in g_i
	quark_image[:,:,:,index]=quark_image[:,:,:,index]-mean_image
	quark_image[:,:,:,index]=quark_image[:,:,:,index]/(std_image+r)




av_quark=np.mean(quark_image,axis=3)

av2_imqfile ='av_image_herwig3_4_5_Color.jpeg'
scipy.misc.imsave(os.path.join(path,av2_imqfile),av_quark)


quark_name='image_herwig3_4_5_color'
gluon_name='image_herwig1_2_color'



    
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6=Process(target=pixel_c.plotting,args=(count1,gluon_image,gluon_name,train_gimage_path,test_gimage_path))
p6.start()
p7=Process(target=pixel_c.plotting,args=(count,quark_image,quark_name,train_qimage_path,test_qimage_path))
p7.start()
p6.join()
p7.join()


#pixel.plotting(gi_gluon,gluon,gluon_name,train_gimage_path)

plt.close()

print (time.time()-start_time), "seconds" 
