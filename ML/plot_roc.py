import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

dt = np.int32

herwigFile1 = "Herwig_fprs_her_200_color.csv"
herwigFile2 = "Herwig_tprs_her_200_color.csv"
herwigFile3 = "Herwig_fprs_pyth_200_color.csv"
herwigFile4 = "Herwig_tprs_pyth_200_color.csv"


pythiaFile1 = "Pythia_fprs_pyth_200_color.csv"
pythiaFile2 = "Pythia_tprs_pyth_200_color.csv"
pythiaFile3 ="Pythia_fprs_her_200_color.csv" #"Pythia_fprs_her_1000.csv"
pythiaFile4 ="Pythia_tprs_her_200_color.csv" #"Pythia_tprs_her_1000.csv"


#HERWIG FILE
#===========#

herwig_fprs_her= pd.read_csv(herwigFile1,dtype=None,delimiter=",")
herwig_fprs_her = np.array(herwig_fprs_her,dtype=float)#

herwig_tprs_her= pd.read_csv(herwigFile2,dtype=None,delimiter=",")
herwig_tprs_her = np.array(herwig_tprs_her,dtype=float)

herwig_fprs_pyth= pd.read_csv(herwigFile3,dtype=None,delimiter=",")
herwig_fprs_pyth=np.array(herwig_fprs_pyth,dtype=float)

herwig_tprs_pyth= pd.read_csv(herwigFile4,dtype=None,delimiter=",")
herwig_tprs_pyth=np.array(herwig_tprs_pyth,dtype=float)


#PYTHIA FILE
#===========#

pythia_fprs_pyth= pd.read_csv(pythiaFile1,dtype=None,delimiter=",")
pythia_fprs_pyth = np.array(pythia_fprs_pyth,dtype=float)

pythia_tprs_pyth= pd.read_csv(pythiaFile2,dtype=None,delimiter=",")
pythia_tprs_pyth = np.array(pythia_tprs_pyth,dtype=float)

pythia_fprs_her= pd.read_csv(pythiaFile3,dtype=None,delimiter=",")
pythia_fprs_her=np.array(pythia_fprs_her,dtype=float)

pythia_tprs_her= pd.read_csv(pythiaFile4,dtype=None,delimiter=",")
pythia_tprs_her=np.array(pythia_tprs_her,dtype=float)

#PLOT
#====#
#ROC

plt.plot([0, 1], [0, 1], '--', color='black')
plt.grid(True)
plt.xlabel('Quark Jet Efficiency(True Positive)')
plt.ylabel('Gluon Jet Rejection(False Negative)')


aucs_py_py = auc(pythia_fprs_pyth, pythia_tprs_pyth, reorder=True)
aucs_py_her = auc(pythia_fprs_her, pythia_tprs_her, reorder=True)

aucs_her_her = auc(herwig_fprs_her, herwig_tprs_her, reorder=True)
aucs_her_py = auc(herwig_fprs_pyth, herwig_tprs_pyth, reorder=True)


plt.ylim((0.0,1.0))
plt.xlim((0.0,1.0))
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Significace Improvement')


plt.plot(pythia_tprs_pyth,1-pythia_fprs_pyth, label='%s (AUC %.2lf)' % ("Pythia CNN on Pythia color images", aucs_py_py))
plt.plot(pythia_tprs_her,1-pythia_fprs_her, label='%s (AUC %.2lf)' % ("Pythia CNN on Herwig color images", aucs_py_her))

plt.plot(herwig_tprs_her,1-herwig_fprs_her, label='%s (AUC %.2lf)' % ("Herwig CNN on Herwig Color Images", aucs_her_her))
plt.plot(herwig_tprs_pyth,1-herwig_fprs_pyth, label='%s (AUC %.2lf)' % ("Herwig CNN on Pythia Color Images", aucs_her_py))



plt.legend(fontsize=10,loc=3)
plt.show()

#SIC

plt.ylim((0.5,3.5))
plt.xlim((0.0,1.0))
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Significace Improvement')


plt.plot(herwig_tprs_her,(herwig_tprs_her/np.sqrt(herwig_fprs_her)),label=("Herwig CNN on Herwig color images"))
plt.plot(herwig_tprs_pyth,(herwig_tprs_pyth/np.sqrt(herwig_fprs_pyth)),label=("Herwig CNN on Pythia color images"))


plt.legend(fontsize=10,loc=3)
plt.show()










