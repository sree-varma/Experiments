import skimage
from skimage import feature
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
import os
import sys
import cv2
import scipy.misc
import scipy as sc
import skimage.transform as sk
import scipy.misc
from multiprocessing import Process,Queue
import  matplotlib.transforms as transforms
from cmath import rect, phase
import skimage.transform as sk
import scipy.misc


radius=input("Radius of the jet: ")
pixel=input("Pixel size:")
number= pixel/2	
delta = (2*radius)/pixel

boxmax=radius
r = 0.00001

def centering(eta,phi,pT):
	"""
	ETA
       -----     
        """
	eta =np.array(eta)
        eta =eta.reshape((np.size(eta),)) 
	mean_eta=np.ma.average(eta,axis=0,weights=pT)
       	eta_new =(eta- mean_eta)
 
	"""
	PHI
       -----     
        """
        phi =np.array(phi) 
        phi =phi.reshape((np.size(phi),))
 	mean_phi = np.ma.average(phi,axis=0,weights=pT)
	phi_new =(phi-mean_phi)		
	
	if (np.sum(np.abs(phi_new)))>boxmax*np.shape(phi_new)[0]:	
		phi_pi = np.zeros((np.shape(phi)[0],))
		for m in range(np.shape(phi)[0]):
			if (phi[m])>=3.13359:
				if (phi[m])<=6.28319:
					phi[m]=phi[m]-2*np.pi
	
		mean_phi_new =np.ma.average(phi,axis=0,weights=pT)
		phi_new=phi-mean_phi_new	
	
	return eta_new,phi_new

def principal_axis(image):
	width, height = image.shape
	pix_coords = np.array([[i, j] for i in range(-width+1, width, 2) for j in range(-height+1, height, 2)])
	covX = np.cov(pix_coords, aweights=np.reshape(image, (width*height)),rowvar=0, bias=1)
	e_vals, e_vecs = np.linalg.eigh(covX)
	pc = e_vecs[:,-1]
	theta = np.arctan2(pc[1], pc[0])
	theta =-90-(theta*180.0/np.pi)
	t_image = sk.rotate(image, theta, order=3)
	pix_bot = np.sum(t_image[:, :-(-height//2)])
	pix_top = np.sum(t_image[:, (height//2):])
	if pix_top > pix_bot:
        	t_image = sk.rotate(t_image, 180.0, order=3)
        	theta += 180.0

	flip=np.fliplr(t_image)
	flip=np.flipud(flip)

	return flip.T





def pixelize(eta_new,phi_new,pT,c2,c3,img_size,number,boxmax,delta,channels=1):
	size=eta_new.shape[0]
	image_array=np.zeros((img_size[0],img_size[1]))
	channel2_array=np.zeros((img_size[0],img_size[1]))
	channel3_array=np.zeros((img_size[0],img_size[1]))

	if channels==1:

       	        for i in range(size):
       	               if (abs(eta_new[i])<=boxmax):				
       	                        j =int(round ((eta_new[i])/delta) )
       	                        j = (j+number)
               	                if (abs(phi_new[i])<=boxmax):
					k = int(round((phi_new[i])/delta))
					k = (k+number)
				
               	                	image_array[j,k]= image_array[j,k]+((pT[i]))

		return image_array
	if channels==3:

       	        for i in range(size):
       	               if (abs(eta_new[i])<=boxmax):				
       	                        j =int(round ((eta_new[i])/delta) )
       	                        j = (j+number)
               	                if (abs(phi_new[i])<=boxmax):
					k = int(round((phi_new[i])/delta))
					k = (k+number)
				
               	                	image_array[j,k]= image_array[j,k]+((pT[i]))
					channel2_array[j,k]= channel2_array[j,k]+((c2[i]))
					channel3_array[j,k]= channel3_array[j,k]+((c3[i]))
		return image_array,channel2_array,channel3_array


def shift(image_array,img_size,number,max_number):
	"""
	Shifting the image to center the global maximum
  
        """
	idx = np.argsort(image_array.ravel())[-max_number:][::-1] 
	max_val = image_array.ravel()[idx]
	maxima = np.c_[np.unravel_index(idx, image_array.shape)]
	#maxima=skimage.feature.peak_local_max(image_array,min_distance=1,num_peaks=3)
	
	maxima=np.asarray(maxima)
	
	#print maxima
	#if maxima.size>0 and maxima.size==6:	
	shift=np.zeros((img_size[0],img_size[1]))	
	shift=sc.ndimage.interpolation.shift(image_array,(number-maxima[0,0],number-maxima[0,1]))


	return shift,maxima


def rot(shift,maxima,img_size,number,max_number):
	"""
	Rotate the image such that the second maximum is in the 12 O'clock position
   
        """

	idx = np.argsort(shift.ravel())[-max_number:][::-1] 
	max_val = shift.ravel()[idx]
	maxima = np.c_[np.unravel_index(idx, shift.shape)]

	#maxima=skimage.feature.peak_local_max(shift,min_distance=1,num_peaks=3)
	maxima=np.asarray(maxima)
	#
	rotate=np.zeros((img_size[0],img_size[1]))
	#print maxima
	#print number-maxima[1,1],number-maxima[1,0]
	
	#if maxima.size==6:
	theta = (np.arctan2(number-maxima[1,1],number-maxima[1,0]))
	rotate=sk.rotate(shift,180-np.rad2deg(theta),order=3)
		
	return rotate,maxima


def flip(rotate,img_size,number,max_number,maxima):
	"""
	Flip the image to ensure the third maximum is in the right half-plane
            
        """

	#if maxima.size==6:
	flip=np.zeros((img_size[0],img_size[1]))

	if(maxima[2,1])<number:
		flip=np.fliplr(rotate)
	else :
		flip=rotate
	
	return flip


def normalize(image,color='grayscale'):
    """Return normalized image array: sum(I) == 1.
    """
    return image / np.sum(image)





def image_array(index_array,df_image,mass_jet,color='grayscale',method='mit'):

	valid_quantities = ['grayscale','color']
	if color is None:
	        print("Please specify the color of the image. Valid choices are:")
	        print(valid_quantities)
	        sys.exit(1)
	elif color not in valid_quantities:
		print("{} is not a valid quantity to extract. Valid types are: ".format(quantity))
        	print(valid_quantities)
        	sys.exit(1)




	global temp1	
	mass = np.zeros(((np.size(index_array)),1),dtype=float)
	jpT = np.zeros(((np.size(index_array)),1),dtype=float)
    	channel1_array=np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
    	channel2_array=np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
    	channel3_array=np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)

	mass_jet=np.array(mass_jet,dtype=float)    
    	mass_jet=mass_jet[:,0]

   
    	#pT_jet=np.array(pT_jet,dtype=float)    
    	#pT_jet=pT_jet[:,0]    

    	zero =np.zeros(1,dtype=float)
    	mass_jet=np.concatenate((zero,mass_jet),axis=0)
    	#pT_jet=np.concatenate((zero,pT_jet),axis=0)
    
    	for index,j in enumerate(index_array):      
        	if index_array[index-1]+1<index_array[index]:
               
                	image= df_image.iloc[index_array[index-1]+1:index_array[index],:]
			image=np.array(image,dtype=float)
			image=image[1:]
		        """
   		         PT
       		       ------     
       		        """
                	pT =np.array(image[:,[2]])
                	pT =pT.reshape((np.size(pT),))
                	pT_sum=np.sum(pT)
		        """
   		        MASS
       		      -------     
       		        
                	mass_p =np.array(image[:,[3]])
                	mass_p =mass_p.reshape((np.size(mass_p),))
                	mass_sum=np.sum(mass_p)


		       
   		        ENERGY
       		       --------     
       		        
                	energy =np.array(image[:,[4]])
                	energy =energy.reshape((np.size(energy),))
                	energy_sum=np.sum(energy)
			"""

			eta_new,phi_new=centering(image[:,[0]],image[:,[1]],pT)
			

		    	if color == 'grayscale':
	        		channel1_array[:,:,index]=pixelize(eta_new,phi_new,pT,pT,pT,img_size=(pixel,pixel),number=number,boxmax=boxmax,delta=delta,channels=1)
				
				if method =='deeptop_18':
					channel1_array[:,:,index]=principal_axis(channel1_array[:,:,index])
				if method=='deeptop_17':

					channel1_array[:,:,index],maxima=shift(channel1_array[:,:,index],img_size=(pixel,pixel),number=number,max_number=3)	
					channel1_array[:,:,index],maxima=rot(channel1_array[:,:,index],maxima,img_size=(pixel,pixel),number=number,max_number=3)
					channel1_array[:,:,index]=flip(channel1_array[:,:,index],img_size=(pixel,pixel),number=number,max_number=3,maxima=maxima)

				if np.sum(channel1_array[:,:,index])==0.0 or np.sum(channel1_array[:,:,index])=='nan' :
					print index,mass_jet[index]   
				if np.sum(channel1_array[:,:,index])!=0.0:
					if method == 'mit' or method== 'deeptop_18':
						channel1_array[:,:,index]=normalize(channel1_array[:,:,index])
					mass[index]= mass_jet[index]
					#jpT[index]= pT_jet[index]
					
	        		
	
    			if color == 'color':
        			channel1_array[:,:,index],channel2_array[:,:,index],channel3_array[:,:,index]=pixelize(eta_new,phi_new,pT,pT,pT,img_size=(pixel,pixel),number=number,boxmax=boxmax,delta=delta,channels=3)

				if np.sum(channel1_array[:,:,index])==0.0 or np.sum(channel2_array[:,:,index])==0.0 or np.sum(channel3_array[:,:,index])==0.0 :
					print index,mass_jet[index]   

				if np.sum(channel1_array[:,:,index])=='nan'or np.sum(channel2_array[:,:,index])=='nan' or np.sum(channel3_array[:,:,index])=='nan' :
					print index,mass_jet[index]   

				if np.sum(channel1_array[:,:,index])!=0.0 and np.sum(channel1_array[:,:,index])!=0.0 and np.sum(channel1_array[:,:,index])!=0.0:
				        
					channel1_array[:,:,index]=normalize(channel1_array[:,:,index])
					#channel3_array[:,:,index]=normalize(channel2_array[:,:,index])
					#channel2_array[:,:,index]=normalize(channel3_array[:,:,index])


					mass[index]= mass_jet[index]
					#jpT[index]= pT_jet[index]
	
    			


        """
         Eliminating empty images
        ---------------------------     
        """
	if color == 'grayscale':
		count=0
		c1_img= np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
		im_mass = np.zeros(((np.size(index_array)),1),dtype=float)
		im_pT = np.zeros(((np.size(index_array)),1),dtype=float)
		for i in range(np.size(index_array)):
			
			if np.sum(channel1_array[:,:,i])!=0:
				c1_img[:,:,count] =channel1_array[:,:,i]
				im_mass[count]=mass[i]
			        im_pT[count] = jpT[i]
		          	count=count+1
	
		c1_img = c1_img[:,:,:count]
		im_mass = im_mass[:count]
		im_pT = im_pT[:count]
		
		image_array= c1_img
		print image_array.shape
	
	
		return image_array,im_mass#,im_pT

	if color == 'color':
		count=0
		c1_img= np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
		c2_img= np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
		c3_img= np.zeros((pixel,pixel,(np.size(index_array))),dtype=float)
		im_mass = np.zeros(((np.size(index_array)),1),dtype=float)
		im_pT = np.zeros(((np.size(index_array)),1),dtype=float)
		for i in range(np.size(index_array)):
			
			if np.sum(channel1_array[:,:,i])!=0 and np.sum(channel2_array[:,:,i])!=0 and np.sum(channel3_array[:,:,i])!=0:
				c1_img[:,:,count] =channel1_array[:,:,i]
				c2_img1[:,:,count] =channel2_array[:,:,i]
				c3_img2[:,:,count] =channel3_array[:,:,i]
				im_mass[count]=mass[i]
			        im_pT[count] = jpT[i]
		          	count=count+1
	
		c1_img = c1_img[:,:,:count]
		c2_img = c2_img[:,:,:count]
		c3_img = c3_img[:,:,:count]
		im_mass = im_mass[:count]
		im_pT = im_pT[:count]
		
		image_array= np.zeros((pixel,pixel,3,count),dtype=float)
		for index in range(count):
			image_array[:,:,:,index]= np.dstack((c1_img[:,:,index],c2_img[:,:,index],c3_img[:,:,index]))
		print image_array.shape
	
	
		return image_array,im_mass#,im_pT



def plotting(index_array,image_array,image_name,train_path,channels=1):#,val_path,test_path):
    if channels==1:   
    	for index in range(index_array): 
		image=image_array[:,:,index]
                imfile ='_%002d.jpeg'%index
		scipy.misc.imsave(os.path.join(train_path,image_name+imfile),image.T)
		
    if channels==3:   
    	for index in range(index_array): 
		image=image_array[:,:,:,index]
                imfile ='_%002d.jpeg'%index
		scipy.misc.imsave(os.path.join(train_path,image_name+imfile),image.T)
























