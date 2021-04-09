# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:59:41 2020

@author: Martin.Sladek
"""


import numpy  as np
import scipy as sp
import pandas as pd
import glob, os
import matplotlib as mpl
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
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


### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)

def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r

# stackoverflow filter outliers - change m as needed (2 is default, 10 filters only most extreme)
def reject_outliers(data, m=10.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

def reject_low(data):
    x = data.Trend.min()
    return data[data.Trend > 10*x]

# Function to slice regularly spaced df of Luminoskan data accroding to n. of tech, biological replicates and treatments
def frame_slice(data, times_col=0, start_col=2, start_sample=0, technical=8, biological=4, treatments=2, group_names=['KO1v', 'KO1dex', 'SAHHv', 'SAHHdex', 'CBS1v', 'CBS1dex', 'CBS2v', 'CBS2dex']):
    list_of_df = []
    multiplier = treatments*biological*technical
    for biol in np.arange(start_sample, start_sample + biological):
        for treat in np.arange(0, treatments):        
            df_list = [data.iloc[0:, times_col]]
            for i, j in [(i, i+8) for i in np.arange(biol*multiplier + treat*technical, (biol+1)*multiplier, treatments*technical)]:
                df_list.append(data.iloc[0:, start_col:].iloc[0:, i:j])
            df = pd.concat(df_list, axis=1)
            df.to_csv(f'{biol}_{treat}.csv')
            list_of_df.append(df)
    output_dict = {}
    for i, v in enumerate(list_of_df):      #returns index, value
        #output_dict["df{0}".format(i)] = v #assign index as key name using .format()
        #output_dict[i] = v                 #use int (from index) as key names instead of string as above
        output_dict[group_names[i]] = v     #use provided list to assign key names
    for k, v in output_dict.items():
        v.to_csv(f'{k}.csv')
    
    
    return list_of_df, output_dict

# Load Luminoskam data
data_raw = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])

# Execute function to slice frame    
list_of_df, output_dict = frame_slice(data_raw)


#Function does not work for HPF MM, need to tweak more or do it manually like this.
technical = 8
biological = 4
treatments = 2
multiplier = treatments*biological*technical
biological2 = 2
multiplier2 = treatments*biological2*technical
signal = data_raw.iloc[0:, 2:]

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





"""
# Dictionary compehension to combine dict with old keys with list to create a new dict with new keys from list
group_names = ['KO1v', 'KO1dex', 'SAHHv', 'SAHHdex', 'CBS1v', 'CBS1dex', 'CBS2v', 'CBS2dex']   
new_dict = {k:v for k in output_dict for k, v in zip(group_names, output_dict.values())}
"""


"""
#example
conc_list = [data_raw.iloc[0:, 0], data_raw.iloc[0:, 2:10]]
final_table = pd.concat(conc_list, axis = 1)

signal = data_raw.iloc[0:, 2:]  # 

#group 1, t1
signal.iloc[0:, 0:8]
signal.iloc[0:, 16:24]
signal.iloc[0:, 32:40]
signal.iloc[0:, 48:56]


#group 1, t2
signal.iloc[0:, 8:16]
signal.iloc[0:, 24:32]
signal.iloc[0:, 40:48]
signal.iloc[0:, 56:64]



#group 2, t1
signal.iloc[0:, 64:72]
signal.iloc[0:, 80:88]
signal.iloc[0:, 96:104]
signal.iloc[0:, 112:120]

#group 2, t2
signal.iloc[0:, 72:80]
signal.iloc[0:, 88:96]
signal.iloc[0:, 104:112]
signal.iloc[0:, 120:128]


g1t1a = [0, 16, 32, 48]  #0 + technical*2
g1t1b = [8, 24, 40, 56]  #technical + techical*2

signal.iloc[0:, g1t1a:g1t1b]

g1t2a = [8, 24, 40, 56]  #technical + technical*2
g1t2b = [16, 32, 48, 64]  #technical+technucal + techical*2, OR [i+8 for i in g1t2a]


g2t1a = [64, 80, 96, 112]
g2ttb = [72, 88, 104, 120]

g2t2a = [72, 88, 104, 120]
g2t2b = [80, 96, 112, 128]
"""


"""
technical = 8
biological = 4
treatments = 2
multiplier = treatments*biological*technical

#create lists of indexes
# group 1, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(0*multiplier + 0*technical, 1*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg1t1 = pd.concat(df_list, axis=1)    

# group 1, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(0*multiplier + 1*technical, 1*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg1t2 = pd.concat(df_list, axis=1)

# group 2, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(1*multiplier + 0*technical, 2*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg2t1 = pd.concat(df_list, axis=1)  

# group 2, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(1*multiplier + 1*technical, 2*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg2t2 = pd.concat(df_list, axis=1)

# group 3, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(2*multiplier + 0*technical, 3*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg3t1 = pd.concat(df_list, axis=1)  

# group 3, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(2*multiplier + 1*technical, 3*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg3t2 = pd.concat(df_list, axis=1)

# group 4, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(3*multiplier + 0*technical, 4*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg4t1 = pd.concat(df_list, axis=1)  

# group 4, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(3*multiplier + 1*technical, 4*multiplier, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg4t2 = pd.concat(df_list, axis=1)


# Special case
biological2 = 2
multiplier2 = treatments*biological2*technical

# group 5, treatment 1
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(4*multiplier, 4*multiplier + multiplier2, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg5t1 = pd.concat(df_list, axis=1)  

# group 5, treatment 2
df_list = [data_raw.iloc[0:, 0]]
for i, j in [(i, i+8) for i in np.arange(4*multiplier + technical, 4*multiplier + multiplier2, treatments*technical)]:
    df_list.append(signal.iloc[0:, i:j])
dfg5t2 = pd.concat(df_list, axis=1)
"""







"""
# CIRCADIAN PARAMETERS

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


# Use amplitude to filter out outliers or nans
#outlier_reindex = ~(np.isnan(reject_outliers(data[['Amplitude']])))['Amplitude']          # need series of bool values for indexing 
outlier_reindex = ~(np.isnan(data['Amplitude']))

data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

phaseseries = data_filt['Phase'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['Unnamed: 0'].values.flatten().astype(int)                      # plot profile name as color
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))     # gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

# LENGTH (AMPLITUDE)
amp = data_filt['Amplitude'].values.flatten()                       # plot filtered Amplitude as length
#amp = 1                                                            # plot arbitrary number if Amp problematic

# POSITION (PHASE)
#phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
#phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using CI or SEM of phase, which is in hours
phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 


###############################################################################################
####### Single Polar Histogram of frequency of phases #########################################
###############################################################################################

N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Histogram_Phase.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Histogram_Phase.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()

"""