import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import math
from matplotlib.colors import LogNorm
import  matplotlib.transforms as transforms
import time
import os
import sys
#from multiprocessing import Process , Queue, Pool
import pixel
import gc
import preprocessing
import argparse

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('-method', '--pre-method', dest='method', type=str, help='Preprocessing methods to be used! mit,deeptop_17,deeptop_18')
parser.add_argument('-dataname', '--image-name', dest='dataname', type=str, help='Name of the dataset to be generated!tW,jW,top,qcd,ZZ,Zj,quarks,gluons')

parser.add_argument('-pt', '--ptrange', dest='ptrange', type=str, help='pt range! 350-400,500-550,1300-1400,500')
parser.add_argument('-R', '--radius', dest='R', type=str, help='Radius of the jet! 0.8,1.5')

parser.add_argument('-channels', '--number of color channels of the image', dest='channels', type=float, default=1, help='Number of color channels of the image')

parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,help='Print output on progress running script')#radius=0.8,pixel=33

args = parser.parse_args()
verbose = args.verbose

method=args.method
dataname=args.dataname

ptrange=args.ptrange
R=args.R

channels=int(args.channels)

r =0.00001

dt=np.float64
start_time = time.time()  
pathim='/data/qcd/data/'+dataname+'/'+ptrange+'/Events/'

path='/data/qcd/data/'

files = folders = 0

for _, dirnames, filenames in os.walk(pathim):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)

if channels==1:
	color="grayscale"
if channels==3:
	color="color"
	

image=[]
mass=[]
def read_csv(filename,path):
    df=pd.read_csv(path+filename,dtype=None,delimiter=",")
    gc.collect()
    return df
for i in range(1,folders+1):

	imFile1='run_%002d/jet_image_R_%s.dat'%(i,R)
	immass1='run_%002d/jet_mass_R_%s.dat'%(i,R)
	if os.path.exists(imFile1) and os.path.exists(immass1) :
		continue
	else:
		imFile1='run_%002d/jet_image.dat'%i
		immass1='run_%002d/jet_mass.dat'%i
			
	
	df=pd.DataFrame([[0,0,0,0]],columns=['ETA','PHI','PT','CHARGE'])
	df_im=read_csv(imFile1,pathim)

	df_mass=pd.read_csv(pathim+immass1,sep='\t',names=["MASS"])

	df1=df.append(df_im.iloc[[-1]],ignore_index=True)
	df_im=df1.append(df_im,ignore_index=True)
	
	
	print ("Read image%002d"%i)
	

	g= df_im.loc[df_im['ETA']=='@@@'] 
	gi = g.index
	a1=np.size(gi)

	im_n1,mass_n1=preprocessing.image_array(gi,df_im,df_mass,color=color,method=method)



	print ("image done!")
	image.append(im_n1)
	mass.append(mass_n1)
	print len(image)


image = np.concatenate([image[i] for i in range(len(image))],axis=2)
mass=np.concatenate([mass[i] for i in range(len(mass))])
np.savez(path+"train_image_"+dataname+"_"+ptrange+"_"+method+".npz",image) 
np.savez(path+"train_mass_"+dataname+"_"+ptrange+"_"+method+".npz",mass)




print (time.time()-start_time), "seconds" 
