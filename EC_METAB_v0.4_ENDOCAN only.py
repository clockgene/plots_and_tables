import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib as mpl

import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from tkinter import filedialog
from tkinter import *

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.api import anova_lm
from statsmodels.graphics.api import interaction_plot, abline_plot

####### Define functions  ###########
#####################################

#to manipulate data (normalize to max value of profil)
def normalize(profilvalues):                                           
    return profilvalues/profilvalues.max()

#to fit with scipy, period=24, degrees of freedom is cca. (n - parameters) = 21-3=18
def cosinor(x, baseline, amplitude, phase):                      
    return baseline + amplitude*np.cos(2*np.pi*(x-phase)/24)

"""    
#alternative cosinor or other periodic function, does not work so far
def cosinor_v2(x, baseline, amplitude, phase, trend, period):   # , damp , period
    #return baseline + amplitude*(np.exp(-damp*x))*np.cos(2*np.pi*(x-phase)/24) + trend*x
    return baseline + amplitude*np.cos(2*np.pi*(x-phase)/period) + trend*x

#horizontal or other basic function for statistics, not used, instead manual calc of mean value line
def horizontal(x):
    return np.mean(x) + 0*x
"""

# to remove negative phase after FFT
def phase_correct(x):    
    if x < -24:
        return x + 48
    elif x < -0:
        return x + 24
    elif x > 120:
        return x - 120    
    elif x > 72:
        return x - 72
    elif x > 48:
        return x - 48
    elif x > 24:
        return x - 24
    else:
        return x

# to solve FFT cosinor sometimes returning (phase-period/2), compares max mean values vs phase/phase+12 and choose 
def phase_correct_2(phase, phase_12, maxv):                         
    if maxv > 12 and phase < 12:
        comp1 = maxv - (phase + 24)
        if np.absolute(comp1) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase
    elif maxv < 12 and phase > 12:
        comp2 = (maxv + 24) - phase
        if np.absolute(comp2) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase
    elif maxv < 12 and phase_12 > 12:
        comp3 = (maxv + 24) - phase_12
        if np.absolute(comp3) < np.absolute(maxv - phase):
            return phase_12
        else:
            return phase
    else:
        if np.absolute(maxv - phase) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase

# to change CT to polar coordinates for polar plotting
# 1h = (2/24)*np.pi = (1/12)*np.pi,   circumference = 2*np.pi*radius
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x-24)/12)*np.pi   
    return r

def array_to_number(array_like):
    if len(array_like) == 0:
        return float(np.nan)
    else:
        return float(array_like)

##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button

root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse folder", command=browse_button)
buttonBrowse.grid()
mainloop()

#############################################################################################################################################

#############################################
######### Original Data Loading #############
#############################################

df_raw = pd.read_csv('.\ENDOCAN.csv', delimiter = ',')
df_raw_i = df_raw.set_index(['tissue', 'Metabolite_name'])
df_raw_u = df_raw_i.stack(level=0)
df_raw_u.index.rename(['tissue', 'Metabolite_name', 'CT'], inplace=True)

df = df_raw_u.unstack(['Metabolite_name']).reset_index()


df.loc[(df['CT'] == "0"), 'CT'] = 0
df.loc[(df['CT'] == "0.1"), 'CT'] = 0
df.loc[(df['CT'] == "0.2"), 'CT'] = 0
df.loc[(df['CT'] == "0.3"), 'CT'] = 0
df.loc[(df['CT'] == "0.4"), 'CT'] = 0
df.loc[(df['CT'] == "4"), 'CT'] = 4
df.loc[(df['CT'] == "4.1"), 'CT'] = 4
df.loc[(df['CT'] == "4.2"), 'CT'] = 4
df.loc[(df['CT'] == "4.3"), 'CT'] = 4
df.loc[df.CT.str.contains("8", na=False), 'CT'] = 8
df.loc[df.CT.str.contains("12", na=False), 'CT'] = 12
df.loc[df.CT.str.contains("16", na=False), 'CT'] = 16
df.loc[df.CT.str.contains("20", na=False), 'CT'] = 20
df.loc[df.CT.str.contains("24", na=False), 'CT'] = 24

df0 = df.copy()

dfh = df_raw_u.reset_index()

dfh.loc[(dfh['CT'] == "0"), 'CT'] = 0
dfh.loc[(dfh['CT'] == "0.1"), 'CT'] = 0
dfh.loc[(dfh['CT'] == "0.2"), 'CT'] = 0
dfh.loc[(dfh['CT'] == "0.3"), 'CT'] = 0
dfh.loc[(dfh['CT'] == "0.4"), 'CT'] = 0
dfh.loc[(dfh['CT'] == "4"), 'CT'] = 4
dfh.loc[(dfh['CT'] == "4.1"), 'CT'] = 4
dfh.loc[(dfh['CT'] == "4.2"), 'CT'] = 4
dfh.loc[(dfh['CT'] == "4.3"), 'CT'] = 4
dfh.loc[dfh.CT.str.contains("8", na=False), 'CT'] = 8
dfh.loc[dfh.CT.str.contains("12", na=False), 'CT'] = 12
dfh.loc[dfh.CT.str.contains("16", na=False), 'CT'] = 16
dfh.loc[dfh.CT.str.contains("20", na=False), 'CT'] = 20
dfh.loc[dfh.CT.str.contains("24", na=False), 'CT'] = 24

dfh.set_index(['tissue', 'Metabolite_name', 'CT'])



################################################
############ Dataframes processing #############
################################################

by1 = dfh.reset_index().groupby('Metabolite_name')
metabolites = []
for a, frame in by1:
    metabolites.append(a)

by2 = dfh.reset_index().groupby('CT')
timepoints = []
for b, frame in by2:
    timepoints.append(b)

by3 = dfh.reset_index().groupby('tissue')
tissues = []
for c, frame in by3:
    tissues.append(c)

categories = ['1.endocannabinoid', '2.NE_NAE', '3.NE_MAG', '4.prostaglandin', '5.thromboxane', '6.leukotriene',
              '7.LA_metabolite', '8.ALA_metabolite', '9.HETE', '10.DHA_metabolite', '11.PUFA']

dfh.loc[(dfh['Metabolite_name'] == "anandamide"), 'category'] = '1.endocannabinoid'
dfh.loc[dfh.Metabolite_name.str.contains("AG"), 'category'] = '1.endocannabinoid'
dfh.loc[dfh.Metabolite_name.str.contains("EA"), 'category'] = '2.NE_NAE'  
dfh.loc[dfh.Metabolite_name.str.contains("glycerol"), 'category'] = '3.NE_MAG'
dfh.loc[(dfh['Metabolite_name'] == "AA"), 'category'] = '4.PUFA'

"""
dfh.loc[dfh.Metabolite_name.str.contains("PG"), 'category'] = '04.prostaglandin'
dfh.loc[dfh.Metabolite_name.str.contains("TXB"), 'category'] = '05.thromboxane'
dfh.loc[dfh.Metabolite_name.str.contains("LTB"), 'category'] = '06.leukotriene'
dfh.loc[dfh.Metabolite_name.str.contains("HOME"), 'category'] = '07.LA_metabolite'
dfh.loc[dfh.Metabolite_name.str.contains("HODE"), 'category'] = '07.LA_metabolite'
dfh.loc[dfh.Metabolite_name.str.contains("HOTrE"), 'category'] = '08.ALA_metabolite'
dfh.loc[dfh.Metabolite_name.str.contains("HETE"), 'category'] = '09.HETE'
dfh.loc[dfh.Metabolite_name.str.contains("HDHA"), 'category'] = '10.DHA_metabolite'
dfh.loc[(dfh['Metabolite_name'] == "DHA"), 'category'] = '11.PUFA'
dfh.loc[(dfh['Metabolite_name'] == "EPA"), 'category'] = '11.PUFA'
"""

dfh.columns = ['tissue', 'Metabolite_name', 'CT', 'levels', 'category']

dfs = dfh.set_index(['tissue', 'category', 'Metabolite_name', 'CT']).sort_index()

df_mean = dfs.groupby(['tissue', 'category', 'Metabolite_name', 'CT']).agg(['mean', 'std', 'sem'])


###############################################
df_mean.to_csv('ENDOCAN_means.csv')
dfs.to_csv('ENDOCAN_out.csv')
df.to_csv('ENDOCAN_raw.csv')
###############################################


###############################################
##### curve fits for all profiles #############
###############################################

######## Initial Guess Parameters #######
p0=[10000, 0, 4.0]     # for cosinor(baseline, amplitude, phase)

####### Prepare lists for data and choose which data to fit #############

df = dfh
df2 = df_mean.reset_index().copy()    # for getting max value for each profile

norm_gene = metabolites
df_cos = pd.DataFrame()
profile = []
Mesor = []
Mesor_SD = []
Amplitude = []
Amp_SD = []
Phase = []
Phase_SD = []
Phase_SEM = []
R2 = []
missing_samples = []
not_expressed = []
horizont = []
Ftest_res = []
pValue = []
maxValue = []
tissue_id = []
cat_id = []
norm_gen_id = []
Peaks = []

####### Fit curve and calculate parameters and compare cos vs horiz by F test ##########

for i in tissues:    

    for norm_gen in norm_gene:
        xdata0 = df.loc[(df['tissue'] == i) & (df['Metabolite_name'] == norm_gen), ['CT']].values.flatten()
        ydata0 = df.loc[(df['tissue'] == i) & (df['Metabolite_name'] == norm_gen), ['levels']].values.flatten()
        newindex = ~((ydata0 == np.nan))
        xdata = xdata0[newindex]            
        ydata = ydata0[newindex]
        #create list of df with CT of max mean expression
        #df2.loc[(df2['tissue'] == i) & (df2['Metabolite_name'] == norm_gen), [('levels', 'mean')]].max()   #this returns max value for each metabolite, need CT of that value
        #df2[('levels', 'mean')] == df2.loc[(df2['tissue'] == i) & (df2['Metabolite_name'] == norm_gen), ('levels', 'mean')].max()     #this returns bool mask from the above
        maxValue.append(df2.loc[(df2[('levels', 'mean')] == df2.loc[(df2['tissue'] == i) & (df2['Metabolite_name'] == norm_gen), ('levels', 'mean')].max()), ['levels', 'CT', 'Metabolite_name', 'category', 'tissue']])                      
        tissue_id.append(i)                 # for MultiIndex
        #cat_id.append(j)                    # for MultiIndex
        norm_gen_id.append(norm_gen)        # for MultiIndex            
                    
        try:                                                    #strange problem when using if/else, this is common workaround to get through NaN
            profile.append(f'{i}_{norm_gen}')
            fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydata, p0, full_output=1)                                        #v1 cosinor curve
            #fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor_v2, xdata, ydata, p0, full_output=1, maxfev=1000000)                        #v2
            ss_res = sum((ydata - cosinor(xdata, fitParams[0], fitParams[1], fitParams[2]))**2)   #residual sum of squares                #v1
            #ss_res = sum((ydata - cosinor_v2(xdata, fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4]))**2)   #residual sum of squares  #v2     
            ss_tot = sum((ydata - np.mean(ydata))**2)       #total sum of squares 
            r_squared = 1 - (ss_res/ss_tot)            
            Mesor.append(float(f'{fitParams[0]}'))
            Amplitude.append(abs(float(f'{fitParams[1]}')))
            Phase.append(phase_correct(float(f'{fitParams[2]}')))                       #add 24 via function, only if phase is negative
            R2.append(float(f'{r_squared}'))
            Mesor_SD.append(np.sqrt(float(f'{fitCovariances[0, 0]}')))                  #np.sqrt(np.diag(fitCovariances)) calculates SD of parameter
            Amp_SD.append(np.sqrt(float(f'{fitCovariances[1, 1]}')))    
            Phase_SD.append(np.sqrt(float(f'{fitCovariances[2, 2]}')))                  #STDEV()/SQRT(COUNT())
            Phase_SEM.append((np.sqrt(float(f'{fitCovariances[2, 2]}')))/np.sqrt(29))    # not sure about N for SEM calc, is Std.Error SEM or SD in prism?
            ss_res_h = ss_tot                                               #horizontal line residuals
            Fvalue = ((ss_res_h - ss_res)/ss_res)/((29-27)/27)
            Ftest_res.append(float(Fvalue))                 #F value - the closer to 1, the more likely simple model is correct and not cosinor, https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?REG_Comparing_Models_Tab.htm
            pVal = (1-stats.f.cdf(Fvalue, 2, 27))           #calculate two-tailed p values of cosinor vs horizontal, formula = (1 - scipy.stats.f.cdf(F,DFnumerator,DFdenominator))
            pValue.append(float(pVal))                      
            x_fit = np.linspace(np.min(xdata), np.max(xdata), 1000)
            y_fit = cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2])
            height = y_fit.min() + (y_fit.max() - y_fit.min())/3
            peak_time, peak_height = find_peaks(y_fit, height=height, distance=24)
            Peaks.append(array_to_number(x_fit[peak_time]))
        except ValueError:
            print(f'{i}_{norm_gen}_is not expressed')
            not_expressed.append(f'{i}_{norm_gen}')
            Mesor.append(np.nan)
            Amplitude.append(np.nan)
            Phase.append(np.nan)   
            R2.append(np.nan)
            Mesor_SD.append(np.nan) 
            Amp_SD.append(np.nan)    
            Phase_SD.append(np.nan) 
            Phase_SEM.append(np.nan)    
            Ftest_res.append(np.nan)  
            pValue.append(np.nan)
            Peaks.append(np.nan)
        except TypeError:
            print(f'{i}_{norm_gen}_is missing')
            #missing_samples.append(f'{i}_{food}')
            missing_samples.append(f'{i}_{norm_gen}')
            Mesor.append(np.nan)
            Amplitude.append(np.nan)
            Phase.append(np.nan)   
            R2.append(np.nan)
            Mesor_SD.append(np.nan)  
            Amp_SD.append(np.nan)    
            Phase_SD.append(np.nan) 
            Phase_SEM.append(np.nan)    
            Ftest_res.append(np.nan)  
            pValue.append(np.nan)
            Peaks.append(np.nan)
        except IndexError:
            Peaks.append(np.nan)

df_cos['profile'] = profile
df_cos['Mesor'] = Mesor
df_cos['Mesor_SD'] = Mesor_SD
df_cos['Amplitude'] = Amplitude
df_cos['Amp_SD'] = Amp_SD
df_cos['Rel_Amp_Error'] = df_cos['Amp_SD']/df_cos['Amplitude']
df_cos['Rel_Amp_SD'] = Amp_SD
df_cos['Phase'] = Phase
df_cos['Phase_SD'] = Phase_SD
df_cos['R2'] = R2
df_cos['P_value'] = pValue
df_cos['Phase_SEM'] = Phase_SEM
df_cos['Phase_12'] = df_cos['Phase'] + 12                                       #alternative Phase for FFT Cosinor problem
df_cos['Peaks'] = Peaks


arrays = [tissue_id, norm_gen_id]                                      #for MultiIndex
index = pd.MultiIndex.from_arrays(arrays, names=('tissue', 'Metabolite_name'))     #for MultiIndex
df_cos2 = df_cos.set_index(index)                                               #this is df_cos with hierarchical multiindex

empty = pd.DataFrame()
df_max_values = empty.append(maxValue)                                          #creates df with CTs and Expr values of max mean of expression
df_max_values2 = df_max_values.drop(columns=[('levels', 'std'), ('levels', 'sem')]).droplevel(1, axis=1)

df_cos_max = (pd.merge(df_max_values2, df_cos2, on=['tissue', 'Metabolite_name']).set_index(['tissue', 'category', 'Metabolite_name']))      # .drop(columns=['Aggr']) #merge cos and max to single dataframe

df_cos_max['Phase_final'] = df_cos_max.apply(lambda x: phase_correct_2(x['Phase'], x['Phase_12'], x['CT']), axis=1).sort_index()     #apply Phase correcting function to remediate FFT Cosinor problem
#df_cos_max.to_csv('df_cos_max.csv')

#####################################################################
######## 1W ANOVA of individual LD or RF profiles for each gene #####
#####################################################################

df = df0.copy().set_index(['tissue', 'CT']).sort_index().fillna(value=0)

not_expressed_s = []
profile_ow = []
P_ow = []
F_ow = []

for i in tissues:    
      
    for norm_gen in norm_gene:
        try:
            profile_ow.append(f'{i}_{norm_gen}')
            model = ols(f'Q("{norm_gen}") ~ C(CT)', df.loc[i, norm_gen][df.loc[i, norm_gen].astype(bool)].reset_index()).fit()    #to overcome patsy formula parsing problem, use Q("var")
            table = anova_lm(model)                             # 1-way ANOVA result
            P_ow.append(table.loc['C(CT)', 'PR(>F)'])           # adds 1w ANOVA p value to list
            F_ow.append(table.loc['C(CT)', 'F'])                # adds 1w ANOVA F value to list
           
        except ValueError:
            print(f'{i}_{norm_gen}_is not expressed')
            not_expressed_s.append(f'{i}_{norm_gen}')
            P_ow.append(np.nan)
            F_ow.append(np.nan)
        except KeyError:
            print(f'{i}_{norm_gen}_is missing')
            not_expressed_s.append(f'{i}_{norm_gen}')
            P_ow.append(np.nan)
            F_ow.append(np.nan)
            

df_stat = pd.DataFrame()
df_stat['profile'] = profile_ow 
df_stat['P_1w_ANOVA'] = P_ow
df_stat['F_1w_ANOVA'] = F_ow
df_stat_i = df_stat.set_index(index)
df_cos_max_stat = pd.merge(df_cos_max.reset_index(), df_stat_i.reset_index(), on=['tissue', 'Metabolite_name']).sort_values(by=['tissue', 'Metabolite_name']).drop(columns=['profile_x', 'profile_y'])
df_cos_max_stat['Rel_Amplitude'] = df_cos_max_stat['Amplitude']/df_cos_max_stat['levels']
df_cos_max_stat['Rel_Amp_SD'] = df_cos_max_stat['Amp_SD']/df_cos_max_stat['levels']

#############################################################################################################################################
df_cos_max_stat.set_index(['tissue', 'category', 'Metabolite_name']).sort_index().to_csv('ENDOCAN_statistics.csv')
#############################################################################################################################################


####### Polar Phase Plot #########################################
##################################################################



####### Single Polar Phase Plot #########################################
# Amplitude is length and width of bar is SD of Phase


#data = df_cos_max_stat.copy().reset_index().set_index(['tissue', 'category']).sort_index().sort_values(by=['Metabolite_name']) 
data = df_cos_max_stat.copy().reset_index().set_index(['tissue', 'category']).sort_values(by=['Metabolite_name']).sort_index()

phaseseries = data.loc['Liver', 'Peaks'].values.flatten()   # use peaks, phase is still quite shitty
phase_sdseries = data.loc['Liver', 'Phase_SD'].values.flatten()
phase = []
phase_sd = []
for phasevalue in phaseseries:
    phase.append(polarphase(phasevalue))
for phasesdvalue in phase_sdseries:
    phase_sd.append(polarphase(phasesdvalue))
amp = data.loc['Liver',  'Rel_Amplitude'].values.flatten()
genes = data.loc['Liver', 'Metabolite_name'].values.flatten()
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)

ax = plt.subplot(111, projection='polar')                                               #plot with polar projection
bars = ax.bar(phase, amp, width=phase_sd, color=colorcode, bottom=0.0, alpha=0.8)       # >prusvitnost-> alpha=0.5
ax.set_ylim(0,0.45)
ax.set_yticks(np.arange(0.1,0.4,0.1))
#ax.set_yticklabels([])          # this deletes radial ticks
ax.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax.set_theta_direction(-1)      #reverse direction of theta increases
ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
ax.legend(bars, genes, fontsize=8, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots 
ax.set_xlabel("Circadian phase (h)", fontsize=12)
plt.title("Rayleigh-style phase plot", fontsize=14, fontstyle='italic')
plt.savefig(f'Liver.svg', format = 'svg', bbox_inches = 'tight')
#plt.show()



