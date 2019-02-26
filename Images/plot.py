import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import math
#from generate import *
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import  matplotlib.transforms as transforms
import time
#import pp
import os
import sys
from multiprocessing import Process , Queue, Pool
import pixel
import gc
import cv2
import scipy.misc
import argparse
import preprocessing
import skimage.transform as sk
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-method', '--pre-method', dest='method', type=str, help='Preprocessing methods to be used! mit,deeptop_17,deeptop_18')

parser.add_argument('-signame', '--signame', dest='signame', type=str, help='Name of the background! tW,top,ZZ')
parser.add_argument('-bgname', '--background-name', dest='bgname', type=str,  help='Name of the background! jW,qcd,Zj')
parser.add_argument('-pt', '--ptrange', dest='ptrange', type=str, help='pt range! 350-400,500-550,1300-1400,500')
parser.add_argument('-R', '--radius', dest='R', type=str, help='Radius of the jet! 0.8,1.5')

parser.add_argument('-channels', '--number of color channels of the image', dest='channels', type=float, default=1, help='Number of color channels of the image')
#parser.add_argument('-steps45', '--zero centering and standardization', dest='zs', type=float, default=1, help='Steps 4-5 in mit paper')
#parser.add_argument('-norm', '--normalization', dest='nor', type=float, default=0, help='Normalization done after plotting average')


parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,
help='Print output on progress running script')

args = parser.parse_args()

verbose = args.verbose
testfiles=input("Is there a test dataset?:  ")
valfiles=input("Is there a validation dataset? : ")
massfiles=input("Massfiles?: " )

method=args.method

signame=args.signame
bgname=args.bgname
ptrange=args.ptrange
R=args.R
r =0.00001
start_time = time.time()

train_bg_image_path='/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/train/'+bgname+'/'

test_bg_image_path='/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/test/'+bgname+'/'


train_sig_image_path='/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/train/'+signame+'/'

test_sig_image_path='/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/test/'+signame+'/'
#test_image_path='/usr/qcd/sample/test/test/'

path='/data/qcd/data/'

trsig_vals=np.load('/data/qcd/data/tW/tW_sample.npz')#np.load(path+"train_image_"+signame+"_"+ptrange+"_"+method+".npz")
sig_image=trsig_vals['arr_0']

trbg_vals=np.load('/data/qcd/data/jW/jW_sample.npz')#np.load(path+"train_image_"+bgname+"_"+ptrange+"_"+method+".npz")
bg_image=trbg_vals['arr_0']
trsig_vals=0
trbg_vals=0

if valfiles==1:
	validation_bg_image_path = '/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/val/'+bgname+'/'
	validation_sig_image_path = '/usr/qcd/mit_mass/'+signame+'_'+bgname+'/'+ptrange+'/R_'+R+'/val/'+signame+'/'
	valsig_vals=np.load(path+"val_image_"+signame+"_"+method+"_.npz")
	valsig_image=valsig_vals['arr_0']

	valbg_vals=np.load(path+"val_image_"+bgname+"_"+method+"_.npz")
	valbg_image=valbg_vals['arr_0']
	valsig_vals=0
	valbg_vals=0

if testfiles==1:
	trainsig_image=sig_image
	trainbg_image=bg_image
	tesig_vals=np.load(path+"test_image_"+signame+"_"+method+"_.npz")
	testsig_image=tesig_vals['arr_0']

	tebg_vals=np.load(path+"test_image_"+bgname+"_"+method+"_.npz")
	testbg_image=tebg_vals['arr_0']
	tesig_vals=0
	tebg_vals=0		

if massfiles==1:
	masssig_vals=np.load('/data/qcd/data/tW/masstW_sample.npz')#np.load(path+"train_mass_"+signame+"_"+ptrange+"_"+method+".npz")
	sig_mass=masssig_vals['arr_0']

	massbg_vals=np.load('/data/qcd/data/jW/massjW_sample.npz')#np.load(path+"train_mass_"+bgname+"_"+ptrange+"_"+method+".npz")
	bg_mass=massbg_vals['arr_0']
	
	train_mass_bg=bg_mass[:int(0.8*bg_mass.shape[0])]
	test_mass_bg =bg_mass[int(0.8*bg_mass.shape[0]):]
	train_mass_sig=sig_mass[:int(0.8*sig_mass.shape[0])]
	test_mass_sig = sig_mass[int(0.8*sig_mass.shape[0]):]
	np.savez(path+"train_massjW_"+ptrange+"_0.8_"+method+"_sample.npz",train_mass_bg)
	np.savez(path+"test_massjW_"+ptrange+"_0.8_"+method+"_sample.npz",test_mass_bg)

	np.savez(path+"train_masstW_"+ptrange+"_0.8_"+method+"_sample.npz",train_mass_sig)
	np.savez(path+"test_masstW_"+ptrange+"_0.8_"+method+"_sample.npz",test_mass_sig)


if testfiles==0:
	
	trainsig_image=sig_image[:,:,:int(0.8*np.shape(sig_image)[2])]
	trainbg_image=bg_image[:,:,:int(0.8*np.shape(bg_image)[2])]
	testsig_image=sig_image[:,:,int(0.8*np.shape(sig_image)[2]):]
	testbg_image=bg_image[:,:,int(0.8*np.shape(bg_image)[2]):]





channels=int(args.channels)
#zs=int(args.zs)
#norm=int(args.nor)


axis=0
if channels==1:
	axis=2

if channels==3:
	axis=3


def plot_images(image,path,name,cbar=plt.cm.seismic,channels=channels,av_image=False):
	if channels==1:
			
		if av_image==True:
			plt.pcolormesh(image.T,cmap =cbar,vmin=-1.0,vmax=1.0)
		else:
				
			plt.pcolormesh(image.T,cmap =cbar)

			#scipy.misc.imsave(os.path.join(path,name), image*(256./np.amax(image)))
		if method=='deeptop_17':
			plt.pcolormesh(image,cmap =cbar,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=np.amin(image),vmax=np.amax(image)))

			plt.xticks([])
			plt.yticks([])
			plt.xlim([0,41])
			plt.ylim([0,41])
			cb = plt.colorbar(ticks=[])
			plt.ylabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
			plt.xlabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
			cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
			plt.grid(True)
			plt.savefig(os.path.join(path,name)) 
			plt.close()	
		if method=='deeptop_18':
			scipy.misc.imsave(os.path.join(path,name), image*(256./np.amax(image)))

		if method=='mit':
			#plt.pcolormesh(image.T,cmap =cbar)
			plt.xticks([])
			plt.yticks([])
			plt.xlim([0,33])
			plt.ylim([0,33])
			cb = plt.colorbar(ticks=[])
			plt.ylabel('Pseudorapidity (normalised) $\eta$', fontsize=16)
			plt.xlabel('Azimuthal Angle (normalised) $\phi$', fontsize=16)
			cb.set_label('Transverse momentum of final state particles in a Jet (GeV)')
			plt.grid(True)
			plt.savefig(os.path.join(path,name)) 
			plt.close()	



	if channels==3:
		scipy.misc.imsave(os.path.join(path,name), image*(256./np.amax(image)))		
	


def av_images(image_array,mean,std,channels=channels):#,q
	if channels==1:
		_,_,index_array=np.shape(image_array)
		for index in range(index_array):
	    		image_array[:,:,index]=(image_array[:,:,index]-mean)/(std+r) 

	if channels==3:
		_,_,_,index_array=np.shape(image_array)
		for index in range(index_array): 
	    		image_array[:,:,:,index]=(image_array[:,:,:,index]-mean)/(std+r)     	



	return image_array
	#q.put(image_array)


train_images = np.concatenate((trainsig_image,trainbg_image),axis=axis)
#train_images1 = np.concatenate((trainsig_img1,trainbg_img1),axis=2)
#val_images = np.concatenate((valsig_image,valbg_image),axis=3)
#test_images =np.concatenate((testsig_image,testbg_image),axis=3)
#images = np.concatenate((train_images,val_images,test_images),axis=3)

mean_sig= np.mean(trainsig_image,axis=axis)
std_sig = np.std(trainsig_image,axis=axis)

mean_bg= np.mean(trainbg_image,axis=axis)
std_bg = np.std(trainbg_image,axis=axis)


mean_image = np.mean(train_images,axis=axis)

std_image = np.std(train_images,axis=axis)




print np.shape(mean_image)
plot_images(mean_image,path,'mean_image.jpeg',cbar=plt.cm.seismic,channels=channels,av_image=False)

print "Mean images creating"

plot_images(mean_sig,path,'av_image_4-5_'+signame+'_'+method+'.jpeg',cbar=plt.cm.jet,channels=channels,av_image=False)
plot_images(mean_bg,path,'av_image_4-5_'+bgname+'_'+method+'.jpeg',cbar=plt.cm.jet,channels=channels,av_image=False)

print "Creating jet images :) "


if method=='mit':
	trainsig_image=av_images(trainsig_image,mean_image,std_image,channels=channels)
	#valsig_image=av_images(valsig_image,mean_image,std_image,channels=channels)
	testsig_image=av_images(testsig_image,mean_image,std_image,channels=channels)

	trainbg_image=av_images(trainbg_image,mean_image,std_image,channels=channels)
	#valbg_image=av_images(valbg_image,mean_image,std_image,channels=channels)
	testbg_image=av_images(testbg_image,mean_image,std_image,channels=channels)
	#sig_images = np.concatenate((trainsig_image,valsig_image),axis=axis)#,testsig_img
	
	av_sig=np.mean(trainsig_image,axis=axis)
	plot_images(av_sig,path,'av_image_'+signame+'_'+method+'.jpeg',cbar=plt.cm.seismic,channels=channels,av_image=True)
	print ("Average top image created")

	
	gc.collect()
	#bg_images = np.concatenate((trainbg_image,valbg_image),axis=axis)#,testbg_img
	av_bg=np.mean(trainbg_image,axis=axis)
	print av_bg

	plot_images(av_bg,path,'av_image_'+bgname+'_'+method+'.jpeg',cbar=plt.cm.seismic,channels=channels,av_image=True)
	gc.collect()
	
	print ("Average qcd image created")


	

gc.collect()



def normalize(image,color='grayscale'):
    """Return normalized image array: sum(I) == 1.
    """
    return image / np.sum(image)




if method=='deeptop_17':
	trainsig_image=normalize(trainsig_image)
	valsig_image=normalize(valsig_image)
	testsig_image=normalize(testsig_image)

	trainbg_image=normalize(trainbg_image)
	valbg_image=normalize(valbg_image)
	testbg_image=normalize(testbg_image)
	#sig_images = normalize(trainsig_image)#,testsig_img	




trsig_name='tr_image_'+signame

trsig_name1='tr_image_'+signame+'_1'
valsig_name='val_image_'+signame
tesig_name='te_image_'+signame

trbg_name='tr_image_'+bgname
valbg_name='val_image_'+bgname
tebg_name='te_image_'+bgname
#test_name='test_image'


_,_,trsig=np.shape(trainsig_image)

#_,_,valsig=np.shape(valsig_image)

_,_,tesig=np.shape(testsig_image)

_,_,trbg=np.shape(trainbg_image)

#_,_,valbg=np.shape(valbg_image)

_,_,tebg=np.shape(testbg_image)




preprocessing.plotting(trsig,trainsig_image,trsig_name,train_sig_image_path,channels=channels)
#pixel.plotting(10,trainsig_img1,trsig_name1,train_sig_image_path)
#trainsig_img=0
#exit()
#preprocessing.plotting(valsig,valsig_image,valsig_name,validation_sig_image_path,channels=channels)
#valsig_img=0
preprocessing.plotting(tesig,testsig_image,tesig_name,test_sig_image_path,channels=channels)
#testsig_img=0
preprocessing.plotting(trbg,trainbg_image,trbg_name,train_bg_image_path,channels=channels)
#trainbg_img=0
#preprocessing.plotting(valbg,valbg_image,valsig_name,validation_bg_image_path,channels=channels)
#valbg_img=0
preprocessing.plotting(tebg,testbg_image,tesig_name,test_bg_image_path,channels=channels)
#testbg_img=0
#pixel.plotting(test,test_img,test_name,test_image_path)

#plt.close()


print (time.time()-start_time), "seconds" 







