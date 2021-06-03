# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:42:24 2021

@author: Martin.Sladek
"""

# imports
import numpy  as np
import scipy as sp
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import winsound
import matplotlib as mpl
import seaborn as sns
import math
import warnings
import glob, os, shutil
from tkinter import filedialog
from tkinter import *
import re
from matplotlib import colors


##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button

# Specify FOLDER
root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse to folder", command=browse_button)
buttonBrowse.grid()
mainloop()
path = os.getcwd() + '\\'

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)


"""
# Line plots from individual wells
data = pd.read_csv(glob.glob(f'{path}*signal.csv')[0])
fig, axs = plt.subplots(16, 24, sharex=True, sharey=True)

counter = 1
yc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
for j in range(24):
    for i in range(16):        
        axs[i, j].plot(data[str(counter)], linewidth=0.1)
        axs[i, j].label_outer()
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticklabels([]) 
        #axs[i, j].set_xlabel(f'{counter}', fontsize=2, labelpad=-5) 
        axs[i, j].set_xlabel(f'{yc[i]}{j + 1} n.{counter}', fontsize=2, labelpad=-10) 
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)
        
        counter += 1

#plt.show() 
#fig.tight_layout()
### To save as bitmap png for easy viewing ###
plt.savefig(f'{path}Composite_Raw_Line_Plot.png', dpi=1000)
### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{path}Composite_Raw_Line_Plot.svg', format = 'svg') #if using rasterized = True to reduce size, set-> dpi = 1000
plt.clf()
plt.close()
"""



# Plot with confidence interval
# first copy names of drugs to signal table and rename signalmod
data1 = pd.read_csv(glob.glob(f'{path}*signalmod.csv')[0])
data1 = data1.set_index('Time')
# use every 8th name only
icols = [j for i,j in enumerate(data1.columns) if i%8==0]
index1 = [val for val in icols for _ in range(8)]
index2 = np.arange(1, 385)
multiindex = list(zip(index1, index2))
mind = pd.MultiIndex.from_tuples(multiindex, names=('drug', 'well'))
data1.columns = mind
data = data1.drop(['no', 'Frame'])

fig, axs = plt.subplots(8, 6, sharex=True, sharey=True)
counter = 0
drugs = index1[0::8]
for j in range(6):
    for i in range(8):
        CI = data[drugs[counter]][12:108].mean(axis=1) + 1.96 * data[drugs[counter]][12:108].std(axis=1)/np.sqrt(len(data[drugs[counter]][12:108].columns))        
        axs[i, j].plot(data.index[12:108], data[drugs[counter]][12:108].mean(axis=1), linewidth=0.1)
        axs[i, j].fill_between(data.index[12:108], (data[drugs[counter]][12:108].mean(axis=1)-CI), (data[drugs[counter]][12:108].mean(axis=1) + CI), color='b', alpha=.1)
        #axs[i, j].set_title(drugs[counter], fontsize=6)
        axs[i, j].label_outer()
        #axs[i, j].set_xticks(data.index[12::24])
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticklabels([])
        #axs[i, j].set_xticklabels(data.index[12::24], fontsize=4) 
        axs[i, j].set_xlabel(f'{drugs[counter]}', fontsize=6, labelpad=-18)  #fontsize=2,        
        axs[i, j].set_yticks([])
        axs[i, j].set_xticks([])
        axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)        
        counter += 1

#fig.tight_layout()
### To save as bitmap png for easy viewing ###
plt.savefig(f'{path}Composite_CI_Line_Plot.png', dpi=1000)
### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{path}Composite_CI_Line_Plot.svg', format = 'svg') #if using rasterized = True to reduce size, set-> dpi = 1000
plt.clf()
plt.close()

"""
# test 1 CI line plot
CI = data['NC'][12:].mean(axis=1) + 1.96 * data['NC'][12:].std(axis=1)/np.sqrt(len(data['NC'][12:].columns))
fig, ax = plt.subplots()
ax.plot(data.index[12:], data['NC'][12:].mean(axis=1))
ax.fill_between(data.index[12:], (data['NC'][12:].mean(axis=1)-CI), (data['NC'][12:].mean(axis=1) + CI), color='b', alpha=.1)
ax.set_xticks(data.index[12::24])
plt.show()
"""