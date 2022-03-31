# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:41:50 2022

@author: Martin.Sladek

v1: load circadian parameters table from per2py and create polar phase plot and histograms
"""

# imports
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import glob, os
from tkinter import filedialog
from tkinter import *

# Use circular colormap (True) or a single color (False) for polar plots? 
circular = False

# Choose color, e.g. 'blue' or 'red', used for linear histograms and for polar plots if circular=False:
color = 'black'


##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button


#############################################################
#############################################################
####### FINAL PLOTS #########################################
#############################################################
#############################################################  

# to change CT to polar coordinates for polar plotting
# 1h = (2/24)*np.pi = (1/12)*np.pi,   circumference = 2*np.pi*radius
# use modulo to get remainder after integer division
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r


# Specify FOLDER
root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse to folder", command=browse_button)
buttonBrowse.grid()
mainloop()
mydir = os.getcwd() + '\\'

    
# LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
# data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
#data_raw = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)


#########################################################################
####### Single Polar Phase Plot #########################################
#########################################################################

# Use amplitude to filter out nans
outlier_reindex = ~(np.isnan(data['Amplitude']))    
data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

# FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
#cols = data_filt.select_dtypes('number').columns   # pick only numeric columns
cols = ['Phase', 'Period', 'Amplitude', 'Decay', 'Rsq','Trend']    # pick hand selected columns
df_sub = data.loc[:, cols]
iqr = df_sub.quantile(0.75, numeric_only=False) - df_sub.quantile(0.25, numeric_only=False)
lim = np.abs((df_sub - df_sub.median()) / iqr) < 2.22
# replace outliers with nan
data_filt.loc[:, cols] = df_sub.where(lim, np.nan)   
# replace outlier-caused nans with median values    
data_filt['Phase'].fillna(data_filt['Phase'].median(), inplace=True)
data_filt['Period'].fillna(data_filt['Period'].median(), inplace=True)
data_filt['Amplitude'].fillna(data_filt['Amplitude'].median(), inplace=True)
data_filt['Decay'].fillna(data_filt['Decay'].median(), inplace=True)
data_filt['Rsq'].fillna(data_filt['Rsq'].median(), inplace=True)
data_filt['Trend'].fillna(data_filt['Trend'].median(), inplace=True)

phaseseries = data_filt['Phase'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['Unnamed: 0'].values.flatten().astype(int)                      # plot profile name as color

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

if circular is True:
    phaseplot_colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))     # gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

else:
    phaseplot_colorcode = color

ax = plt.subplot(111, projection='polar')                                                       #plot with polar projection
bars = ax.bar(phase, amp, width=phase_sd, color=phaseplot_colorcode, bottom=0, alpha=0.8)       #transparency-> alpha=0.5, , rasterized = True, bottom=0.0 to start at center, bottom=amp.max()/3 to start in 1/3 circle
#ax.set_yticklabels([])          # this deletes radial ticks
ax.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax.set_theta_direction(-1)      #reverse direction of theta increases
ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
#ax.legend(bars, genes, fontsize=8, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots 
ax.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Invidual phases plot", fontsize=14, fontstyle='italic')


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Phase plot.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Phase plot.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()



###############################################################################################
####### Single Polar Histogram of frequency of phases with Rayleigh vector#####################
###############################################################################################

N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
#colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html
#colorcode = sns.husl_palette(256)[0::int(round(len(colors) / N_bins, 0))]
#colorcode = colors[0::int(round(len(colors) / N_bins, 0))] 
if circular is True:
    polar_histogram_colorcode = sns.husl_palette(256)[0::int(round(len(sns.husl_palette(256)) / N_bins, 0))]
else:
    polar_histogram_colorcode = color
    
phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=polar_histogram_colorcode, bottom=1, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')

# calculate vector sum of angles and plot "Rayleigh" vector
a_cos = map(lambda x: math.cos(x), phase)
a_sin = map(lambda x: math.sin(x), phase)
uv_x = sum(a_cos)/len(phase)
uv_y = sum(a_sin)/len(phase)
uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
uv_phase = np.angle(complex(uv_x, uv_y))

# Alternative from http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
#v_angle = math.atan((uv_y/uv_radius)/(uv_x/uv_radius))

v_angle = uv_phase     # they are the same 
v_length = uv_radius*max(phase_hist)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black')) #add arrow

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Histogram_Phase.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Histogram_Phase.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()



###############################################################################################
####### Single Histogram of frequency of periods and phases ###################################
###############################################################################################

#outlier_reindex_per = ~(np.isnan(reject_outliers(data[['Period']])))['Period'] 
#data_filt_per = data_filt[outlier_reindex_per]

data_filt_per = data_filt.copy()

######## Single Histogram ##########
y = "Period"
x_lab = y
y_lab = "Counts"
ylim = (0, 0.4)
xlim = (math.floor(data_filt['Period'].min() - 1), math.ceil(data_filt_per['Period'].max() + 1))
suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)

allplot = sns.FacetGrid(data_filt_per)    
#plots PDF when kde=True, can be >1, https://stats.stackexchange.com/questions/4220/can-a-probability-distribution-value-exceeding-1-be-ok
allplot = allplot.map(sns.distplot, y, kde=False, color=color)  #, bins=n, bins='sqrt' for Square root of n, None for Freedman–Diaconis rule
plt.xlim(xlim)
#plt.legend(title='Sex')
plt.xlabel(x_lab)
plt.ylabel(y_lab)
nmean = round(data_filt_per[y].mean(), 3)
plt.text(x_coord, y_coord, f'n = {str(data_filt_per[y].size - data_filt_per[y].isnull().sum())}\nmean = {nmean} ± {str(round(data_filt_per[y].sem(), 3))}h')
#loc = plticker.MultipleLocator(base=4.0) # this locator puts ticks at regular intervals
#allplot.xaxis.set_major_locator(loc)

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}' + '\\' + 'Histogram_Period.svg', format = 'svg', bbox_inches = 'tight')
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}' + '\\' + 'Histogram_Period.png', format = 'png', bbox_inches = 'tight')
plt.clf()
plt.close()

######## Single Histogram ##########
y = "Phase"
x_lab = y
y_lab = "Counts"
ylim = (0, 0.4)
xlim = (0, 360)
suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)


allplot = sns.FacetGrid(data_filt_per)    
#plots PDF when kde=True, can be >1, https://stats.stackexchange.com/questions/4220/can-a-probability-distribution-value-exceeding-1-be-ok
allplot = allplot.map(sns.distplot, y, kde=False, color=color)  #, bins=n, bins='sqrt' for Square root of n, None for Freedman–Diaconis rule
plt.xlim(xlim)
#plt.legend(title='Sex')
plt.xlabel(x_lab)
plt.ylabel(y_lab)
plt.text(x_coord, y_coord, f'n = ' + str(data_filt_per[y].size - data_filt_per[y].isnull().sum()) + '\nmean = ' + str(round(data_filt_per[y].mean(), 3)) + ' ± ' + str(round(data_filt_per[y].sem(), 3)) + 'h')
#loc = plticker.MultipleLocator(base=4.0) # this locator puts ticks at regular intervals
#allplot.xaxis.set_major_locator(loc)

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}' + '\\' + 'Histogram_Phase_lin.svg', format = 'svg', bbox_inches = 'tight')
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}' + '\\' + 'Histogram_Phase_lin.png', format = 'png', bbox_inches = 'tight')
plt.clf()
plt.close()
