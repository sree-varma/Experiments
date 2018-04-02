import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
#from collections import defaultdict
import time
#import pp
import os
import sys
#from multiprocessing import Process,Queue
import matplotlib.image
import scipy.misc


delta = 0.8/33 # length of each pixel
boxmax=0.4
r = 0.00001
total_pT = 0.00
#q = Queue()
#temp1 = np.zeros((33,33,5001),dtype=float)
def image_array(index_array,image_array,df_image,q):
    global temp1

    for index,j in enumerate(index_array): # For all values and their corresponding indices in g_i
	
        #print "for all values and their corresponding indices in g_i"
        if index_array[index-1]+1<index_array[index]:
                #print "if there are vales between index"
                image= df_image.iloc[index_array[index-1]+1:index_array[index],:]
                image= np.array(image[1:],dtype=float)
                #print image
                  #PT
                #=======#
                pT =np.array(image[:,[2]])# Call the third column in all image file as 'pT'
                
                pT =pT.reshape((np.size(pT),))# Reshape it for plotting otherwise errors
                #print np.sum(pT)
                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta =np.array(image[:,[0]])# Call the first column in all image file as 'eta'
                eta =eta.reshape((np.size(eta),)) # Reshape it for plotting otherwise errors
                mean_eta = np.ma.average(eta,axis=0,weights=pT)# Weighted mean     \

                eta =(eta- mean_eta) # Centering
                size = np.size(eta)# Number of particles in the jet size(eta)=size(phi)=size(pT)

                  #PHI
                #=======#
                phi =np.array(image[:,1]) # Call the second column in all image file as 'phi'
                phi_image =phi.reshape((np.size(phi),)) # Reshape it for plotting otherwise errors
                mean_phi = np.ma.average(phi,axis=0,weights=pT)#Weighted mean
                phi =(phi-mean_phi)#Centering
                   #PIXEL
                #=======#
                for i in range(0,size):
                        #print "image_array kk munne olla loop"
                        if (abs(eta[i])<=boxmax):#boxmax): # if the value of -0.4<=eta<=0.
                                # print "eta 0.4 nekkal cheruthayal"
                                # image_array[16,16,index]=pT[0] # The center pixel is JET PT
                                j = int(round(((eta[i])/delta))) # a number between -16 to 16
                                j = j+16 #j+16 = 16 if j =0 => The central pixel value
				#j = j(dtype= 'int')
                                if (abs(phi[i])<=boxmax): # a number between -16 to 16
                                        #print "phi 0.4 nekkal cheruthayal"
                                        k = int(round(((phi[i])/delta)))
                                        k = k+16 #k+16 = 16 if k =0 => The central pixel value.
					#print k
					#k = k(dtype= 'int')
                                        image_array[j,k,index]= image_array[j,k,index]+pT[i]#image is an 3D array, each layer gives an image
                                        #f.write(str(image[j,k,index])+'\n')

                total_pT=np.sum(image_array[:,:,index])# Total pT in an image
                #print total_pT
		if total_pT==0 or total_pT=='nan':
			image_array[:,:,index] = image_array[:,:,index]/(total_pT+0.00001)
                #print total_pT
		else:
                	image_array[:,:,index] = image_array[:,:,index]/total_pT #Normalisation
    temp1=image_array        #return image_array
    #print temp1[:,16,1]
    q.put(temp1)
    #return temp1 #image_array


def image_2array(index_array,image_array,df_image):
    global temp2

    for index,j in enumerate(index_array): # For all values and their corresponding indices in g_i
        #print "for all values and their corresponding indices in g_i"
        if index_array[index-1]+1<index_array[index]:
                #print "if there are vales between index"
                image= df_image.iloc[index_array[index-1]+1:index_array[index],:]
                image= np.array(image[1:],dtype=float)
                #print image
                  #PT
                #=======#
                pT =np.array(image[:,[2]])# Call the third column in all image file as 'pT'
                pT =pT.reshape((np.size(pT),))# Reshape it for plotting otherwise errors

                        #f.write(str(pT[0])+'\n')
                  #ETA
                #=======#
                eta =np.array(image[:,[0]])# Call the first column in all image file as 'eta'
                eta =eta.reshape((np.size(eta),)) # Reshape it for plotting otherwise errors
                mean_eta = np.average(eta,axis=0,weights=pT)# Weighted mean     \

                eta =(eta- mean_eta) # Centering
                size = np.size(eta)# Number of particles in the jet size(eta)=size(phi)=size(pT)

                  #PHI
                #=======#
                phi =np.array(image[:,1]) # Call the second column in all image file as 'phi'
                phi_image =phi.reshape((np.size(phi),)) # Reshape it for plotting otherwise errors
                mean_phi = np.average(phi,axis=0,weights=pT)#Weighted mean
                phi =(phi-mean_phi)#Centering
                   #PIXEL
                #=======#
                for i in range(0,size):
                        #print "image_array kk munne olla loop"
                        if (abs(eta[i])<=boxmax):#boxmax): # if the value of -0.4<=eta<=0.
                                # print "eta 0.4 nekkal cheruthayal"
                                # image_array[16,16,index]=pT[0] # The center pixel is JET PT
                                j = int (round(((eta[i])/delta))) # a number between -16 to 16
                                j = j+16 #j+16 = 16 if j =0 => The central pixel value
                                if (abs(phi[i])<=boxmax): # a number between -16 to 16
                                        #print "phi 0.4 nekkal cheruthayal"
                                        k = int (round(((phi[i])/delta)))
                                        k = k+16 #k+16 = 16 if k =0 => The central pixel value
                                        image_array[j,k,index]= image_array[j,k,index]+pT[i]#image is an 3D array, each layer gives an image
                                        #f.write(str(image[j,k,index])+'\n')

                total_pT=np.sum(image_array)# Total pT in an image
                image_array[:,:,index] = image_array[:,:,index]/total_pT #Normalisation
    temp2=image_array        #return image_array
    #return temp2#image_array





def jet_images(index_array,x,y,image_array,mean,std):
    for index,j in enumerate(index_array): # For all values and their corresponding indices in g_i
        if index_array[index-1]+1<index_array[index]:
            for k in range(0,x):
                for j in range(0,y):
                    image_array[j,k,index]=(image_array[j,k,index]-mean[j,k]) #Step 4 & 5: Zerocentering and Standardizing
                    image_array[j,k,index]=image_array[j,k,index]/(std[j,k]+0.00001) #Step5: Standardize
    return image_array


          

def shuffle(image_array):
    x,y,z = image_array.shape
    zero_array=np.zeros((x,y,z),dtype=float)
    a= np.arange(0,z)
    b=np.random.shuffle(a)
    for i in range(0,z):
        zero_array[:,:,a[i]]=zero_array[:,:,a[i]]+image_array[:,:,i]

    return zero_array




def plotting(index_array,image_array,image_name,train_path,test_path):
    for index in range(0,index_array): # For all values and their corresponding indices in gi_quark
        #if index_array[index-1]+1<index_array[index]:
                # Plotting
                #=========#
                image_array_new=image_array[:,:,index]
                #plt.rcParams['axes.facecolor'] = 'white'
                #fig_size = plt.rcParams["figure.figsize"]
                #fig_size[0] = 0.428571428
                #fig_size[1] = 0.428571428
                #plt.figure(frameon=False)
                #plt.axis('off')
                #plt.rcParams["figure.figsize"] = fig_size
                #plt.axes().set_aspect('equal')
                #plt.pcolormesh(image_array_new,cmap=plt.cm.jet,vmin=-1.0,vmax=+1.0)
                #plt.xlim([0,33])
                #plt.ylim([0,33])
                #plt.xticks([])
                #plt.yticks([])#
		cmap =plt.get_cmap('jet')
                if index<100001:
                        imfile ='_%002d.jpeg'%index# Name image files as image_01.png
                        #plt.savefig(os.path.join(train_path,image_name+imfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        #plt.close()
			#matplotlib.image.imsave(os.path.join(train_path,image_name+imfile),image_array_new,cmap=cmap)
                        scipy.misc.imsave(os.path.join(train_path,image_name+imfile),image_array_new)

                else:
                        imfile ='_%002d.jpeg'%index# Name image files as image_01.png
                        #plt.savefig(os.path.join(test_path,image_name+imfile),bbox_inches='tight', pad_inches = 0) # Save image files in the given folder
                        #plt.close()
			#matplotlib.image.imsave(os.path.join(test_path,image_name+imfile),image_array_new,cmap=cmap)
 			scipy.misc.imsave(os.path.join(test_path,image_name+imfile),image_array_new)

                        
    plt.close()

