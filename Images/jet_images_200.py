import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
#import pp
import os
import sys
from multiprocessing import Process , Queue
import pixel
import matplotlib.image


start_time = time.time()  # Time to execute the program.
path=''# Path of the program

#PATH OF FILES
#--------------
#1.Quarks
#--------

pathq1='/usr/Herwig/data/'
pathq2='/usr/Herwig/data/'  # Path of the program
pathq3='/usr/Herwig/data/'


#2.Gluons
#---------
pathg1 = '/usr/Herwig/data/'
pathg2='/usr/Herwig/data/'  # Path of the program

#3.Images
#--------
train_qimage_path='/usr/Herwig/images_200-220GeV/train/quarks/'#'/home/k1629656/Jets/herwig_100/train/quarks'#'/home/k1629656/Jets/train/quarks/'#'/usr/Herwig/gluon_jets/images_200-220GeV'#\
test_qimage_path='/usr/Herwig/images_200-220GeV/test/quarks/'#'/home/k1629656/Jets/herwig_100/test/quarks'#'/home/k1629656/Jets/test/quarks/'


train_gimage_path='/usr/Herwig/images_200-220GeV/train/gluons/' #'/home/k1629656/Jets/train/gluons/'#'/usr/Herwig/gluon_jets/images_200-220GeV'#\
test_gimage_path='/usr/Herwig/images_200-220GeV/test/gluons/' #'/home/k1629656/Jets/test/gluons/'


#FILES
#======
#1.QUARKS
#--------

quarkFile1='herwig_gg2qqbar_40000_200-220_jet_image.dat'  # The generated data f\rom Herwig simulation
quarkFile2='herwig_qq2qq_40000_200-220_jet_image.dat'
quarkFile3='herwig_qqbar2qqbar_40000_200-220_jet_image.dat'
df_quark1 = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
df_quark2 = pd.read_csv(pathq2+quarkFile2,dtype=None,delimiter=",")
df_quark3 = pd.read_csv(pathq3+quarkFile3,dtype=None,delimiter=",")
df_quark = pd.concat([df_quark1,df_quark2,df_quark3],ignore_index=True,axis=0)


#df_quark.to_csv('herwig_quark_jet_120000_200-220_jet_image.dat',index = False)
#df_quark=df_quark.iloc[:,:]#68193,:]#2650000,:]#1344527,:]#2650000,:]
gquark= df_quark.loc[df_quark['ETA']=='@@@'] 
gi_quark = gquark.index # Indices of '@@@'#

#print gi_quark[3000]

#113442,:]#[:679365,:]#:1344527:]
#df_quark2=df_quark.iloc[679365:1344527,:]#:2650000,:] # index of the last line in 125000th Jet


#pathq1= '/usr/Pythia/quark_jets/'
#quarkFile1='pythia_quark_jet_30000_200-220_jet_image.dat'
#df_quark = pd.read_csv(pathq1+quarkFile1,dtype=None,delimiter=",")
#df_quark=df_quark.iloc[:129994,:]#:129966,:]#208976,:]#62709,:]

#2.GLUONS
#---------

gluonFile1='herwig_gg2gg_60000_200-220_jet_image.dat'  # The generated data from Herwig simulation
gluonFile2 ='herwig_qqbar2gg_60000_200-220_jet_image.dat'
df_gluon1 = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
df_gluon2 = pd.read_csv(pathg2+gluonFile2,dtype=None,delimiter=",")

df_gluon = pd.concat([df_gluon1,df_gluon2],ignore_index=True,axis=0)
#df_gluon.to_csv('herwig_gloun_jet_120000_200-220_jet_image.dat',index = False)
#df_gluon=df_gluon.iloc[:,:]#80665,:]#3182541,:]
ggluon= df_gluon.loc[df_gluon['ETA']=='@@@'] 
gi_gluon = ggluon.index # Indices of '@@@'#

#df_quark1=df_quark1.iloc[:22932,:]
#df_quark2=df_quark2.iloc[:21866,:]
#df_quark3=df_quark3.iloc[:212361,:]
#df_gluon1=df_gluon1.iloc[:40623,:]
#df_gluon2=df_gluon2.iloc[:38841,:]




#print gi_gluon[3000]
#df_gluon= pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
#df_gluon=df_gluon.iloc[:1611689,:]

#134568,:]#[:807656,:]#[:1611689,:]
#df_gluon2= df_gluon.iloc[807656:1611689,:]#[1611689:3182541,:] # index of the last line in 125000th Jet


#pathg1= '/usr/Pythia/gluon_jets/'
#gluonFile1='pythia_gloun_jet_30000_200-220_jet_image.dat'
#df_gluon = pd.read_csv(pathg1+gluonFile1,dtype=None,delimiter=",")
#df_gluon=df_gluon.iloc[:196393,:]#298560,:]#89633,:]
#print df_quark1.head()
gquark1= df_quark1.loc[df_quark1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark1 = gquark1.index # Indices of '@@@'#
gquark2= df_quark2.loc[df_quark2['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark2 = gquark2.index # Indices of '@@@'#
gquark3= df_quark3.loc[df_quark3['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_quark3 = gquark3.index # Indices of '@@@'#


ggluon1= df_gluon1.loc[df_gluon1['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_gluon1 = ggluon1.index # Indices of '@@@'
ggluon2= df_gluon2.loc[df_gluon2['ETA']=='@@@'] # The locations where the events end which is given in the simulation as '@@@'
gi_gluon2 = ggluon2.index # Indices of '@@@'

#print gi_quark1[1000],gi_quark2[1000],gi_quark3[1000]

#print gi_gluon1[1500],gi_gluon2[1500]



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

quark1= np.zeros((33,33,a1),dtype=float)
quark2= np.zeros((33,33,a2),dtype=float)
quark3= np.zeros((33,33,a3),dtype=float)
gluon1= np.zeros((33,33,b1),dtype=float)
gluon2= np.zeros((33,33,b2),dtype=float)

print ("Read quark and gluon files")


if __name__=='__main__':
    q1 = Queue()#
    p1=Process(target=pixel.image_array,args=(gi_quark1,quark1,df_quark1,q1))
    p1.start()
    q2 = Queue()#
    p2=Process(target=pixel.image_array,args=(gi_quark2,quark2,df_quark2,q2))
    p2.start()
    q3 = Queue()#
    p3=Process(target=pixel.image_array,args=(gi_quark3,quark3,df_quark3,q3))
    p3.start()         
    g1= Queue()
    p4=Process(target=pixel.image_array,args=(gi_gluon1,gluon1,df_gluon1,g1))
    p4.start()
    g2= Queue()
    p5=Process(target=pixel.image_array,args=(gi_gluon2,gluon2,df_gluon2,g2))
    p5.start()
   

quark1=q1.get()
quark1=np.array(quark1).reshape((33,33,a1))
quark2=q2.get()
quark2=np.array(quark2).reshape((33,33,a2))
quark3=q3.get()
quark3=np.array(quark3).reshape((33,33,a3))
print "Quark image array created!"

gluon1=g1.get()
gluon1=np.array(gluon1).reshape((33,33,b1))
gluon2=g2.get()
gluon2=np.array(gluon2).reshape((33,33,b2))
print "Gluon image array created!"


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()


#QUARKS#
#quark=pixel.image_array(gi_quark,q1,df_quark)


#gluon=pixel.image_array(gi_gluon,g1,df_gluon)

#x,y,z=np.shape(gluon)
#print x,y,z
#print quark.shape
#print gluon.shape

quark01=np.zeros((33,33,(a1+a2+a3)),dtype=float)
quark01[:,:,:a1]=quark1
quark01[:,:,a1:a1+a2]=quark2
quark01[:,:,a1+a2:a1+a2+a3]=quark3

count=0
quark= np.zeros((33,33,(a1+a2+a3)),dtype=float)
print quark.shape
for i in range(0,(a1+a2+a3)):
	if np.sum(quark01[:,:,i])!=0:
		quark[:,:,count] = quark01[:,:,i]
		count=count+1
		
print count
quark = quark[:,:,:count]

quark =pixel.shuffle(quark)
print quark.shape


gluon01=np.zeros((33,33,(b1+b2)),dtype=float)
gluon01[:,:,:b1]=gluon1
gluon01[:,:,b1:b1+b2]=gluon2
x,y,z=np.shape(gluon01)

count1=0
gluon= np.zeros((33,33,(b1+b2)),dtype=float)
print gluon.shape
for i in range(0,(b1+b2)):
	if np.sum(gluon01[:,:,i])!=0:
		gluon[:,:,count1] = gluon01[:,:,i]
		count1=count1+1
		
print count1
gluon = gluon[:,:,:count1]


gluon=pixel.shuffle(gluon)
print gluon.shape

train_quark = quark[:,:,:100000]
train_gluon = gluon[:,:,:100000]

images=np.zeros((33,33,(200000)),dtype=float)
images[:,:,:100000]=quark[:,:,:100000]
images[:,:,100000:]=gluon[:,:,:100000]


mean_image = np.mean(images,axis=2)

std_image = np.std(images,axis=2)
print std_image
images=0

#Plotting average image before zero centering and standardizing
#==============================================================#
mean_quark= np.mean(quark[:,:,:100000],axis=2)#Average of all images
std_quark= np.std(quark[:,:,:100000],axis=2)#Std of all images


mquark = pd.DataFrame(mean_quark)
mquark.to_csv("mean_quark_gif.txt")

mean_gluon=np.mean(gluon[:,:,:100000],axis=2)
std_gluon=np.std(gluon[:,:,:100000],axis=2)

mgluons = pd.DataFrame(mean_gluon)
mgluons.to_csv("mean_gluons_gif.txt")

mean=np.zeros((33,33,200000),dtype=float)#mean=np.zeros((33,33,2),dtype=float)#
mean[:,:,:100000]=train_quark#mean[:,:,0]=mean_quark#
mean[:,:,100000:200000]=train_gluon#mean[:,:,1]=mean_gluon#

std=np.zeros((33,33,200000),dtype=float)#std=np.zeros((33,33,200000),dtype=float)
std[:,:,:100000]=train_quark#std[:,:,0]=std_quark#
std[:,:,100000:200000]=train_gluon#std[:,:,1]=std_gluon#

#mean_image=np.mean(mean,axis=2)
#std_image=np.std(std,axis=2)
p,q =np.shape(mean_image)
image =np.zeros((p,q),dtype=float)
for i in range (0,p):
    for j in range(0,q):
        image[i,j]=(mean_image[i,j]/(std_image[i,j]+r)) 

plt.pcolormesh(image,cmap =plt.cm.bwr)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#v = np.arange(qminpt,qmaxpt)
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imqfile ='image.jpeg'
plt.savefig(os.path.join(path,av1_imqfile)) # Save image files in the given fol\

plt.close()




#mimage = pd.DataFrame(mean_image)
#mimage.to_csv("mean_image_gif.txt")
#simage = pd.DataFrame(std_image)
#simage.to_csv("std_image_gif.txt")
print "Mean images creating"



qminpt=np.amin(mean)
qmaxpt=np.amax(mean)
#print minpt
#print maxpt
plt.pcolormesh(mean_quark,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#v = np.arange(qminpt,qmaxpt)
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imqfile ='av_image_before_step4-5_herwig3_4_5_.jpeg'
plt.savefig(os.path.join(path,av1_imqfile)) # Save image files in the given folder
plt.close()








gminpt=np.amin(mean_gluon)
gmaxpt=np.amax(mean_gluon)
#print minpt
#print maxpt
plt.pcolormesh(mean_gluon,cmap =plt.cm.jet)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#v = np.arange(gminpt,gmaxpt)
v=np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av1_imgfile ='av_image_before_step4-5_herwig1_2_.jpeg'
plt.savefig(os.path.join(path,av1_imgfile)) # Save image files in the given folder
plt.close()

print "Creating jet images :) "


#gluon1=pixel.jet_images(gi_gluon,x,y,gluon,mean_image,std_image)


for index in range(0,count1): # For all values and their corresponding indices in g_i
	gluon[:,:,index]=gluon[:,:,index]-mean_image #Step 4 : Zerocentering
	gluon[:,:,index]=gluon[:,:,index]/(std_image+r) #Step5: Standardize


av_gluon=np.mean(gluon,axis=2)
#av_gluons = pd.DataFrame(av_gluon)
#av_gluons.to_csv("av_gluons_gif.txt")







print av_gluon

mgminpt=np.amin(av_gluon)
mgmaxpt=np.amax(av_gluon)
plt.rcParams['axes.facecolor'] = 'white'
plt.axes().set_aspect('equal')
fig_size = plt.rcParams["figure.figsize"]
plt.figure(frameon=False)
plt.axis('off')


plt.pcolormesh(av_gluon,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)
v = np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#cb = plt.colorbar()
#plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
#plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
#cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av2_imgfile ='av_image_herwig1_2.jpeg'#herwig1_2.jpeg'
plt.savefig(os.path.join(path,av2_imgfile)) # Save image files in the given folder
plt.close()
av_gluons = pd.DataFrame(av_gluon)
av_gluons.to_csv("av_gluons_gif.txt")

mimage = pd.DataFrame(mean_image)
mimage.to_csv("mean_image_gif.txt")
simage = pd.DataFrame(std_image)
simage.to_csv("std_image_gif.txt")






for index in range(0,count): # For all values and their corresponding indices in g_i
	quark[:,:,index]=quark[:,:,index]-mean_image #Step 4 : Zerocentering
	quark[:,:,index]=quark[:,:,index]/(std_image+r) #Step5: Standardize





av_quark=np.mean(quark,axis=2)
print av_quark
mqminpt=np.amin(av_quark)
mqmaxpt=np.amax(av_quark)
plt.rcParams['axes.facecolor'] = 'white'
plt.axes().set_aspect('equal')
fig_size = plt.rcParams["figure.figsize"]
#fig_size[0] = 10
#fig_size[1] = 10
plt.figure(frameon=False)
plt.axis('off')

plt.pcolormesh(av_quark,cmap =plt.cm.bwr,vmin=-1.0,vmax=1.0)

v = np.arange(-1.0,1.0)
cb = plt.colorbar(ticks=v)
plt.xticks([])
plt.yticks([])
plt.xlim([0,33])
plt.ylim([0,33])
#cb = plt.colorbar()
#plt.xlabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
#plt.ylabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
#cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
plt.grid(True)
av2_imqfile ='av_image_herwig3_4_5.jpeg'#herwig3_4_5.jpeg'
plt.savefig(os.path.join(path,av2_imqfile)) # Save image files in the given folder
plt.close()



#for index,j in enumerate(gi_gluon): # For all values and their corresponding indices in gi_quark
#        if gi_gluon[index-1]+1<gi_gluon[index]:
                # Plotting
                #=========#
#                gluon_new=gluon[:,:,index]
#                plt.rcParams['axes.facecolor'] = 'white'
#                fig_size = plt.rcParams["figure.figsize"]
#                fig_size[0] = 0.4155844155
#                fig_size[1] = 0.4155844155
#                plt.figure(frameon=False)
#                plt.axis('off')
#                plt.rcParams["figure.figsize"] = fig_size
#                plt.axes().set_aspect('equal')
#                plt.pcolormesh(gluon_new,cmap=plt.cm.jet,vmin=-1.0,vmax=+1.0)
#                plt.xlim([0,33])
#                plt.ylim([0,33])
#                plt.xticks([])
#                plt.yticks([])#

#                if index<100001:
#                        gimfile ='image_herwig1_2_%002d.jpeg'%index# Name image files as image_01.png
#                        plt.savefig(os.path.join(train_gimage_path,gimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
#                        plt.close()
#                else:
#                        gimfile ='image_herwig1_2_%002d.jpeg'%index# Name image files as image_01.png
#                        plt.savefig(os.path.join(test_gimage_path,gimfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
#                        plt.close()
                        

quark_name='image_herwig3_4_5'
gluon_name='image_herwig1_2'



#if __name__=='__main__':
    
#p1.join()
#p2.join()
#p3.join()
#p4.join()
#p5.join()
p6=Process(target=pixel.plotting,args=(count1,gluon,gluon_name,train_gimage_path,test_gimage_path))
p6.start()
p7=Process(target=pixel.plotting,args=(count,quark,quark_name,train_qimage_path,test_qimage_path))
p7.start()
p6.join()
p7.join()


#pixel.plotting(gi_gluon,gluon,gluon_name,train_gimage_path)

plt.close()


print (time.time()-start_time), "seconds" 
