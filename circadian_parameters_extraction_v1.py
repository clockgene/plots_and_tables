# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:12:44 2021
v.20211102
@author: Martin.Sladek
extract all parameters from all subfolders and put them to common dataframe and export as csv, for Prism plots
"""
import pandas as pd
import glob, os
import tkinter as tk
from tkinter import filedialog

# was the experiment without treatment (treat = 0) or with treatment (treat = 1)?
treat = 1

# Set name and nameend variables, depends on length of folder names
name = -10               # -4 for nontreated CP, -5 for nontreated SCN, for treated CP -8, for treated SCN -12
nameend = None          # for treated use None, othewise try -4


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
    df[f'Period' + str(mydir[name:nameend])] = data['Period']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df[f'Amplitude' + str(mydir[name:nameend])] = data['Amplitude']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df[f'Phase' + str(mydir[name:nameend])] = data['Phase']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df[f'Trend' + str(mydir[name:nameend])] = data['Trend']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df[f'Decay' + str(mydir[name:nameend])] = data['Decay']
for mydir in mydirlist:
    data = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
    df[f'Rsq' + str(mydir[name:nameend])] = data['Rsq']


df.to_csv(f'{path}Composite_parameters.csv')