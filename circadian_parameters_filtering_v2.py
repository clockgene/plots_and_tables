# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:59:41 2020
Script contains function to slice results from luminoskan to different groups for easy plottong in Prism.
Also at the bottom is another script for cirk. param. table from per2py, this is just separated manually to different csvs.
@author: Martin.Sladek
"""


import numpy  as np
import pandas as pd
import glob, os
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog


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
mydir = os.getcwd() + '\\'


# Function to slice regularly spaced df of Luminoskan data accroding to n. of tech, biological replicates and treatments
# tech replicates = rows, biol replicates = columns, treatment = groups of rows (e.g. 2 treatments on row 1-8 and 9-16)
# times_col - where is time index, start_col - where the data start, start_sample - from where to apply slicing
# group_names - names for all groups, separately for all treatmens, use default numbers, or load names.csv file
def frame_slice(data, group_names=np.arange(1, 49), times_col=0, start_col=2, start_sample=0, technical=8, biological=4, treatments=2):
    list_of_df = []
    multiplier = treatments*biological*technical
    for biol in np.arange(start_sample, start_sample + biological):
        for treat in np.arange(0, treatments):        
            df_list = [data.iloc[0:, times_col]]
            for i, j in [(i, i+8) for i in np.arange(biol*multiplier + treat*technical, (biol+1)*multiplier, treatments*technical)]:
                df_list.append(data.iloc[0:, start_col:].iloc[0:, i:j])
            df = pd.concat(df_list, axis=1)            
            list_of_df.append(df)
    # to assign names to dfs, create dict, then use it to save dfs as named csv files
    output_dict = {}
    for i, v in enumerate(list_of_df):      #returns index, value
        #output_dict["df{0}".format(i)] = v #assign index as key name using .format()
        #output_dict[i] = v                 #use int (from index) as key names instead of string as above
        output_dict[group_names[i]] = v     #use provided list to assign key names
    for k, v in output_dict.items():
        v.to_csv(f'{k}.csv')       
    return list_of_df, output_dict


# Execute function to slice frame, load csv file names.csv with group names in columns on row 0 (may have more names but need to be in correct order) 
list_of_df, output_dict = frame_slice(pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0]), group_names=pd.read_csv(glob.glob(f'{mydir}*names.csv')[0]).columns)


#Function does not work for nonregular items (e.g. here HPF MM), need to do it manually like this:
data_raw = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])
technical = 8
biological = 4
treatments = 2
multiplier = treatments*biological*technical
biological2 = 2
multiplier2 = treatments*biological2*technical
signal = data_raw.iloc[0:, 2:]
# group 5, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(4*multiplier, 4*multiplier + multiplier2, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
MMv = pd.concat(df_list, axis=1)  
MMv.to_csv(f'MMv.csv')
# group 5, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(4*multiplier + technical, 4*multiplier + multiplier2, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
MMdex = pd.concat(df_list, axis=1)
MMdex.to_csv(f'MMdex.csv')


# to select part of data for plotting..., call dict values using keys, like this
#output_dict['CBS2dex']

"""
# Dictionary compehension to combine dict with old keys with list to create a new dict with new keys from list
group_names = ['KO1v', 'KO1dex', 'SAHHv', 'SAHHdex', 'CBS1v', 'CBS1dex', 'CBS2v', 'CBS2dex']   
new_dict = {k:v for k in output_dict for k, v in zip(group_names, output_dict.values())}
"""


"""
# CIRCADIAN PARAMETERS slicing, manually assign filters using XY coordinates from params.csv

data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])

data_ko_v = data.loc[(data['X'] < 5) & (data['Y'] < 9)]
data_ko_dex = data.loc[(data['X'] < 5) & (data['Y'] > 8)]
data_sahh_v = data.loc[(data['X'] > 4) & (data['X'] < 9) & (data['Y'] < 9)]
data_sahh_dex = data.loc[(data['X'] > 4) & (data['X'] < 9) & (data['Y'] > 8)]
data_cbs1_v = data.loc[(data['X'] > 8) & (data['X'] < 13) & (data['Y'] < 9)]
data_cbs1_dex = data.loc[(data['X'] > 8) & (data['X'] < 13) & (data['Y'] > 8)]
data_cbs2_v = data.loc[(data['X'] > 12) & (data['X'] < 17) & (data['Y'] < 9)]
data_cbs2_dex = data.loc[(data['X'] > 12) & (data['X'] < 17) & (data['Y'] > 8)]
data_mm_v = data.loc[(data['X'] > 16) & (data['X'] < 19) & (data['Y'] < 9)]
data_mm_dex = data.loc[(data['X'] > 16) & (data['X'] < 19) & (data['Y']  > 8)]

data_ko_v.to_csv(f'{mydir}data_ko_v.csv')
data_ko_dex.to_csv(f'{mydir}data_ko_dex.csv')
data_sahh_dex.to_csv(f'{mydir}data_sahh_dex.csv')
data_sahh_v.to_csv(f'{mydir}data_sahh_v.csv')
data_cbs1_v.to_csv(f'{mydir}data_cbs1_v.csv')
data_cbs1_dex.to_csv(f'{mydir}data_cbs1_dex.csv')
data_cbs2_dex.to_csv(f'{mydir}data_cbs2_dex.csv')
data_cbs2_v.to_csv(f'{mydir}data_cbs2_v.csv')
data_mm_v.to_csv(f'{mydir}data_mm_v.csv')
data_mm_dex.to_csv(f'{mydir}data_mm_dex.csv')
"""
