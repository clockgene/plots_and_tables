# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:25:58 2022
@author: Martin.Sladek
v.2022.04.22
remove outliers from violin plots
"""

import pandas as pd
import glob, os
import tkinter as tk
from tkinter import filedialog
import numpy  as np

import seaborn as sns
import matplotlib.pyplot as plt
#import scipy.stats as stats
#import scikit_posthocs as sp

# was the experiment without treatment (treat = 0) or with treatment (treat = 1)?
treat = 1

# Set name and nameend variables, depends on length of folder names
name = -9               # -4 for nontreated CP, -5 for nontreated SCN, for treated CP -8, for treated SCN -12
nameend = None          # None, or try -4
    
##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button

root = tk.Tk()
folder_path = tk.StringVar()
lbl1 = tk.Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = tk.Button(text="Browse folder", command=browse_button)
buttonBrowse.grid()
tk.mainloop()
path = os.getcwd() + '\\'
#path = os.getcwd()

depth = path.count(os.sep)
paths = [path]

mydirlist = []
for root, dirs, files in os.walk(path, topdown=False):
    for files in dirs:        
        folder = os.path.join(root, files)
        #mydir = os.path.abspath(directories)
        #if folder.count(os.sep) == depth + 1:
        if folder.count(os.sep) == depth + treat:
            mydirlist.append(folder)

df = pd.DataFrame()

for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Period' + str(mydir[name:nameend])] = data['Period']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Amplitude' + str(mydir[name:nameend])] = data['Amplitude']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Phase' + str(mydir[name:nameend])] = data['Phase']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Trend' + str(mydir[name:nameend])] = data['Trend']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Decay' + str(mydir[name:nameend])] = data['Decay']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df['Rsq' + str(mydir[name:nameend])] = data['Rsq']


df.to_csv(f'{path}Composite_parameters.csv')
ticks = [i[name+1:nameend] for i in mydirlist]


# Violin plot from wide-format dataframe 
def violin(data, title, ticks=ticks, remove_outliers=True):
    title = title               
    ax = sns.violinplot(data=data)
    plt.title(title, fontsize=14)
    ax.axes.xaxis.set_ticklabels(ticks)
    plt.xticks(rotation=90)
    plt.savefig(f'{path}Violin_{title}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'{path}Violin_{title}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()

i = len(mydirlist)

violin(df.iloc[:, 2*i:3*i], 'Phase')

# For other parameters, first replace outliers with nans
for col in df.columns.values:
    # FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lim = np.abs((df[col] - df[col].median()) / iqr) < 2.22
    df.loc[:, col] = df[col].where(lim, np.nan)

violin(df.iloc[:, 0:i], 'Period')
violin(df.iloc[:, i:2*i], 'Amplitude')
violin(df.iloc[:, 3*i:4*i], 'Trend')
violin(df.iloc[:, 4*i:5*i], 'Decay')
violin(df.iloc[:, 5*i:6*i], 'Rsq')
