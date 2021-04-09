import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

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
import math


##################### Tkinter button for browse to workdir ################################
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


#################################################################################################################

# custom colors for nice circular hues
circular_colors = np.array([[0.91510904, 0.55114749, 0.67037311],
   [0.91696411, 0.55081563, 0.66264366],
   [0.91870995, 0.55055664, 0.65485881],
   [0.92034498, 0.55037149, 0.64702356],
   [0.92186763, 0.55026107, 0.63914306],
   [0.92327636, 0.55022625, 0.63122259],
   [0.9245696 , 0.55026781, 0.62326754],
   [0.92574582, 0.5503865 , 0.6152834 ],
   [0.92680349, 0.55058299, 0.6072758 ],
   [0.92774112, 0.55085789, 0.59925045],
   [0.9285572 , 0.55121174, 0.59121319],
   [0.92925027, 0.551645  , 0.58316992],
   [0.92981889, 0.55215808, 0.57512667],
   [0.93026165, 0.55275127, 0.56708953],
   [0.93057716, 0.5534248 , 0.55906469],
   [0.93076407, 0.55417883, 0.55105838],
   [0.93082107, 0.55501339, 0.54307696],
   [0.93074689, 0.55592845, 0.53512681],
   [0.9305403 , 0.55692387, 0.52721438],
   [0.93020012, 0.55799943, 0.51934621],
   [0.92972523, 0.55915477, 0.51152885],
   [0.92911454, 0.56038948, 0.50376893],
   [0.92836703, 0.56170301, 0.49607312],
   [0.92748175, 0.56309471, 0.48844813],
   [0.9264578 , 0.56456383, 0.48090073],
   [0.92529434, 0.56610951, 0.47343769],
   [0.92399062, 0.56773078, 0.46606586],
   [0.92254595, 0.56942656, 0.45879209],
   [0.92095971, 0.57119566, 0.4516233 ],
   [0.91923137, 0.5730368 , 0.44456642],
   [0.91736048, 0.57494856, 0.4376284 ],
   [0.91534665, 0.57692945, 0.43081625],
   [0.91318962, 0.57897785, 0.42413698],
   [0.91088917, 0.58109205, 0.41759765],
   [0.90844521, 0.58327024, 0.41120533],
   [0.90585771, 0.58551053, 0.40496711],
   [0.90312676, 0.5878109 , 0.3988901 ],
   [0.90025252, 0.59016928, 0.39298143],
   [0.89723527, 0.5925835 , 0.38724821],
   [0.89407538, 0.59505131, 0.38169756],
   [0.89077331, 0.59757038, 0.37633658],
   [0.88732963, 0.60013832, 0.37117234],
   [0.88374501, 0.60275266, 0.36621186],
   [0.88002022, 0.6054109 , 0.36146209],
   [0.87615612, 0.60811044, 0.35692989],
   [0.87215369, 0.61084868, 0.352622  ],
   [0.86801401, 0.61362295, 0.34854502],
   [0.86373824, 0.61643054, 0.34470535],
   [0.85932766, 0.61926872, 0.3411092 ],
   [0.85478365, 0.62213474, 0.3377625 ],
   [0.85010767, 0.6250258 , 0.33467091],
   [0.84530131, 0.62793914, 0.3318397 ],
   [0.84036623, 0.63087193, 0.32927381],
   [0.8353042 , 0.63382139, 0.32697771],
   [0.83011708, 0.63678472, 0.32495541],
   [0.82480682, 0.63975913, 0.32321038],
   [0.81937548, 0.64274185, 0.32174556],
   [0.81382519, 0.64573011, 0.32056327],
   [0.80815818, 0.6487212 , 0.31966522],
   [0.80237677, 0.65171241, 0.31905244],
   [0.79648336, 0.65470106, 0.31872531],
   [0.79048044, 0.65768455, 0.31868352],
   [0.78437059, 0.66066026, 0.31892606],
   [0.77815645, 0.66362567, 0.31945124],
   [0.77184076, 0.66657827, 0.32025669],
   [0.76542634, 0.66951562, 0.3213394 ],
   [0.75891609, 0.67243534, 0.32269572],
   [0.75231298, 0.67533509, 0.32432138],
   [0.74562004, 0.6782126 , 0.32621159],
   [0.73884042, 0.68106567, 0.32836102],
   [0.73197731, 0.68389214, 0.33076388],
   [0.72503398, 0.68668995, 0.33341395],
   [0.7180138 , 0.68945708, 0.33630465],
   [0.71092018, 0.69219158, 0.33942908],
   [0.70375663, 0.69489159, 0.34278007],
   [0.69652673, 0.69755529, 0.34635023],
   [0.68923414, 0.70018097, 0.35013201],
   [0.6818826 , 0.70276695, 0.35411772],
   [0.67447591, 0.70531165, 0.3582996 ],
   [0.667018  , 0.70781354, 0.36266984],
   [0.65951284, 0.71027119, 0.36722061],
   [0.65196451, 0.71268322, 0.37194411],
   [0.64437719, 0.71504832, 0.37683259],
   [0.63675512, 0.71736525, 0.38187838],
   [0.62910269, 0.71963286, 0.38707389],
   [0.62142435, 0.72185004, 0.39241165],
   [0.61372469, 0.72401576, 0.39788432],
   [0.60600841, 0.72612907, 0.40348469],
   [0.59828032, 0.72818906, 0.40920573],
   [0.59054536, 0.73019489, 0.41504052],
   [0.58280863, 0.73214581, 0.42098233],
   [0.57507535, 0.7340411 , 0.42702461],
   [0.5673509 , 0.7358801 , 0.43316094],
   [0.55964082, 0.73766224, 0.43938511],
   [0.55195081, 0.73938697, 0.44569104],
   [0.54428677, 0.74105381, 0.45207286],
   [0.53665478, 0.74266235, 0.45852483],
   [0.52906111, 0.74421221, 0.4650414 ],
   [0.52151225, 0.74570306, 0.47161718],
   [0.5140149 , 0.74713464, 0.47824691],
   [0.506576  , 0.74850672, 0.48492552],
   [0.49920271, 0.74981912, 0.49164808],
   [0.49190247, 0.75107171, 0.4984098 ],
   [0.48468293, 0.75226438, 0.50520604],
   [0.47755205, 0.7533971 , 0.51203229],
   [0.47051802, 0.75446984, 0.5188842 ],
   [0.46358932, 0.75548263, 0.52575752],
   [0.45677469, 0.75643553, 0.53264815],
   [0.45008317, 0.75732863, 0.5395521 ],
   [0.44352403, 0.75816207, 0.54646551],
   [0.43710682, 0.758936  , 0.55338462],
   [0.43084133, 0.7596506 , 0.56030581],
   [0.42473758, 0.76030611, 0.56722555],
   [0.41880579, 0.76090275, 0.5741404 ],
   [0.41305637, 0.76144081, 0.58104704],
   [0.40749984, 0.76192057, 0.58794226],
   [0.40214685, 0.76234235, 0.59482292],
   [0.39700806, 0.7627065 , 0.60168598],
   [0.39209414, 0.76301337, 0.6085285 ],
   [0.38741566, 0.76326334, 0.6153476 ],
   [0.38298304, 0.76345681, 0.62214052],
   [0.37880647, 0.7635942 , 0.62890454],
   [0.37489579, 0.76367593, 0.63563704],
   [0.37126045, 0.76370246, 0.64233547],
   [0.36790936, 0.76367425, 0.64899736],
   [0.36485083, 0.76359176, 0.6556203 ],
   [0.36209245, 0.76345549, 0.66220193],
   [0.359641  , 0.76326594, 0.66873999],
   [0.35750235, 0.76302361, 0.67523226],
   [0.35568141, 0.76272903, 0.68167659],
   [0.35418202, 0.76238272, 0.68807086],
   [0.3530069 , 0.76198523, 0.69441305],
   [0.35215761, 0.7615371 , 0.70070115],
   [0.35163454, 0.76103888, 0.70693324],
   [0.35143685, 0.76049114, 0.71310742],
   [0.35156253, 0.75989444, 0.71922184],
   [0.35200839, 0.75924936, 0.72527472],
   [0.3527701 , 0.75855647, 0.73126429],
   [0.3538423 , 0.75781637, 0.73718884],
   [0.3552186 , 0.75702964, 0.7430467 ],
   [0.35689171, 0.75619688, 0.74883624],
   [0.35885353, 0.75531868, 0.75455584],
   [0.36109522, 0.75439565, 0.76020396],
   [0.36360734, 0.75342839, 0.76577905],
   [0.36637995, 0.75241752, 0.77127961],
   [0.3694027 , 0.75136364, 0.77670417],
   [0.37266493, 0.75026738, 0.7820513 ],
   [0.37615579, 0.74912934, 0.78731957],
   [0.37986429, 0.74795017, 0.79250759],
   [0.38377944, 0.74673047, 0.797614  ],
   [0.38789026, 0.74547088, 0.80263746],
   [0.3921859 , 0.74417203, 0.80757663],
   [0.39665568, 0.74283455, 0.81243022],
   [0.40128912, 0.74145908, 0.81719695],
   [0.406076  , 0.74004626, 0.82187554],
   [0.41100641, 0.73859673, 0.82646476],
   [0.41607073, 0.73711114, 0.83096336],
   [0.4212597 , 0.73559013, 0.83537014],
   [0.42656439, 0.73403435, 0.83968388],
   [0.43197625, 0.73244447, 0.8439034 ],
   [0.43748708, 0.73082114, 0.84802751],
   [0.44308905, 0.72916502, 0.85205505],
   [0.44877471, 0.72747678, 0.85598486],
   [0.45453694, 0.72575709, 0.85981579],
   [0.46036897, 0.72400662, 0.8635467 ],
   [0.4662644 , 0.72222606, 0.86717646],
   [0.47221713, 0.72041608, 0.87070395],
   [0.47822138, 0.71857738, 0.87412804],
   [0.4842717 , 0.71671065, 0.87744763],
   [0.4903629 , 0.71481659, 0.88066162],
   [0.49649009, 0.71289591, 0.8837689 ],
   [0.50264864, 0.71094931, 0.88676838],
   [0.50883417, 0.70897752, 0.88965898],
   [0.51504253, 0.70698127, 0.89243961],
   [0.52126981, 0.70496128, 0.8951092 ],
   [0.52751231, 0.70291829, 0.89766666],
   [0.53376652, 0.70085306, 0.90011093],
   [0.54002912, 0.69876633, 0.90244095],
   [0.54629699, 0.69665888, 0.90465565],
   [0.55256715, 0.69453147, 0.90675397],
   [0.55883679, 0.69238489, 0.90873487],
   [0.56510323, 0.69021993, 0.9105973 ],
   [0.57136396, 0.68803739, 0.91234022],
   [0.57761655, 0.68583808, 0.91396258],
   [0.58385872, 0.68362282, 0.91546336],
   [0.59008831, 0.68139246, 0.91684154],
   [0.59630323, 0.67914782, 0.9180961 ],
   [0.60250152, 0.67688977, 0.91922603],
   [0.60868128, 0.67461918, 0.92023033],
   [0.61484071, 0.67233692, 0.921108  ],
   [0.62097809, 0.67004388, 0.92185807],
   [0.62709176, 0.66774097, 0.92247957],
   [0.63318012, 0.66542911, 0.92297153],
   [0.63924166, 0.66310923, 0.92333301],
   [0.64527488, 0.66078227, 0.92356308],
   [0.65127837, 0.65844919, 0.92366082],
   [0.65725076, 0.65611096, 0.92362532],
   [0.66319071, 0.65376857, 0.92345572],
   [0.66909691, 0.65142302, 0.92315115],
   [0.67496813, 0.64907533, 0.92271076],
   [0.68080311, 0.64672651, 0.92213374],
   [0.68660068, 0.64437763, 0.92141929],
   [0.69235965, 0.64202973, 0.92056665],
   [0.69807888, 0.6396839 , 0.91957507],
   [0.70375724, 0.63734122, 0.91844386],
   [0.70939361, 0.63500279, 0.91717232],
   [0.7149869 , 0.63266974, 0.91575983],
   [0.72053602, 0.63034321, 0.91420578],
   [0.72603991, 0.62802433, 0.9125096 ],
   [0.7314975 , 0.62571429, 0.91067077],
   [0.73690773, 0.62341425, 0.9086888 ],
   [0.74226956, 0.62112542, 0.90656328],
   [0.74758193, 0.61884899, 0.90429382],
   [0.75284381, 0.6165862 , 0.90188009],
   [0.75805413, 0.61433829, 0.89932181],
   [0.76321187, 0.6121065 , 0.89661877],
   [0.76831596, 0.6098921 , 0.89377082],
   [0.77336536, 0.60769637, 0.89077786],
   [0.77835901, 0.6055206 , 0.88763988],
   [0.78329583, 0.6033661 , 0.88435693],
   [0.78817477, 0.60123418, 0.88092913],
   [0.79299473, 0.59912616, 0.87735668],
   [0.79775462, 0.59704339, 0.87363986],
   [0.80245335, 0.59498722, 0.86977904],
   [0.8070898 , 0.592959  , 0.86577468],
   [0.81166284, 0.5909601 , 0.86162732],
   [0.81617134, 0.5889919 , 0.8573376 ],
   [0.82061414, 0.58705579, 0.85290625],
   [0.82499007, 0.58515315, 0.84833413],
   [0.82929796, 0.58328538, 0.84362217],
   [0.83353661, 0.58145389, 0.83877142],
   [0.8377048 , 0.57966009, 0.83378306],
   [0.8418013 , 0.57790538, 0.82865836],
   [0.84582486, 0.57619119, 0.82339871],
   [0.84977422, 0.57451892, 0.81800565],
   [0.85364809, 0.57289   , 0.8124808 ],
   [0.85744519, 0.57130585, 0.80682595],
   [0.86116418, 0.56976788, 0.80104298],
   [0.86480373, 0.56827749, 0.79513394],
   [0.86836249, 0.56683612, 0.789101  ],
   [0.87183909, 0.56544515, 0.78294645],
   [0.87523214, 0.56410599, 0.77667274],
   [0.87854024, 0.56282002, 0.77028247],
   [0.88176195, 0.56158863, 0.76377835],
   [0.88489584, 0.56041319, 0.75716326],
   [0.88794045, 0.55929505, 0.75044023],
   [0.89089432, 0.55823556, 0.74361241],
   [0.89375596, 0.55723605, 0.73668312],
   [0.89652387, 0.55629781, 0.72965583],
   [0.89919653, 0.55542215, 0.72253414],
   [0.90177242, 0.55461033, 0.71532181],
   [0.90425   , 0.55386358, 0.70802274],
   [0.90662774, 0.55318313, 0.70064098],
   [0.90890408, 0.55257016, 0.69318073],
   [0.91107745, 0.55202582, 0.68564633],
   [0.91314629, 0.55155124, 0.67804225]])


#continous color maps
#cmap="viridis"
#cmap="YlGnBu"
#cmap= grayscale_cmap(cmap)
#other circular color maps
#cmap = mpl.colors.ListedColormap(sns.hls_palette(256))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256, .33, .85, .6))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256))
#cmap = mpl.colors.ListedColormap(circular_colors)

# for larger figures, need to make the lines thinner
mpl.rcParams['axes.linewidth'] = 0.5

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)


#############################################
######### Original Data Loading #############
#############################################



#data_ArB = pd.read_csv('biocyc_ArB.csv')  # all amplitudes
data_ArB = pd.read_csv('biocyc_ArB_hiamp.csv')  # amplitudes > 1
#data_BrA = pd.read_csv('biocyc_BrA.csv')
data_BrA = pd.read_csv('biocyc_BrA_hiamp.csv')
data_dge = pd.read_csv('DGE_high.csv')
data_ArB12 = pd.read_csv('biocyc12_ArB.csv')
data_BrA12 = pd.read_csv('biocyc12_BrA.csv')
data_ArB_counts24 = pd.read_csv('biocyc_ArB_counts24.csv')
data_BrA_counts24 = pd.read_csv('biocyc_BrA_counts24.csv')
data_fpkm = pd.read_csv('fpkm_filtered.csv')

print(data_ArB.head())
print(data_BrA.head())

"""
# normalize each column with fpkm values, needed for hiamp data, which is not prenormalized
data_ArB = data_ArB.apply(lambda x: x/x.max() if x.name in data_ArB.iloc[:, 2:].columns else x, axis=0)
data_BrA = data_BrA.apply(lambda x: x/x.max() if x.name in data_BrA.iloc[:, 2:].columns else x, axis=0)


#####################################
####### Heatmap #####################
#####################################

##### prepare plot style ##########
sns.set_context("paper")   # , font_scale=1.2
#sns.set_palette("husl")
sns.set_palette("Set1", 8, .75)
#sns.set_palette("Set3", 10)
sns.set_style("ticks")  # "ticks", "white"

cmap='viridis_r'
#cmap='YlGnBu'

### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)
# causes problem with rotation of ylabel ? #


###### Heatmap 1 ############
suptitle1 = "HiAmp Transcripts rhythmic in Sham/ad lib"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
# ArB heat data

df_heat_spec2 = data_ArB.iloc[:, 2:].loc[data_ArB['group'] == 'A'].transpose()
df_heat_spec3 = data_ArB.iloc[:, 2:].loc[data_ArB['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
#heat_SHAM = sns.heatmap(df_heat_spec2, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SHAM = sns.heatmap(df_heat_spec2, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec3, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap="YlGnBu"

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('HiAmpArB.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()


###### Heatmap2 ############
suptitle2 = "HiAmp Transcripts rhythmic in SCNx/tRF"

###### Heatmap plot ############

# BrA heat data

df_heat_spec0 = data_BrA.iloc[:, 2:].loc[data_BrA['group'] == 'A'].transpose()
df_heat_spec1 = data_BrA.iloc[:, 2:].loc[data_BrA['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
#heat_SHAM = sns.heatmap(df_heat_spec0, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SHAM = sns.heatmap(df_heat_spec0, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 

heat_SCNx = sns.heatmap(df_heat_spec1, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap) 

fig.suptitle(suptitle2, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.savefig('HiAmpBrA.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()

"""
#####################################

"""
###### Heatmap 3 ############
suptitle1 = "Differentially expressed genes"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
# ArB heat data

df_heat_spec4 = data_dge.iloc[:, 2:].loc[data_dge['group'] == 'A'].transpose()
df_heat_spec5 = data_dge.iloc[:, 2:].loc[data_dge['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(df_heat_spec4, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=True, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec5, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap="YlGnBu"

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('DGE.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()





###### Heatmap 12a ############
suptitle1 = "Transcripts with 12h period in Sham/ad lib"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
# ArB heat data

df_heat_spec6 = data_ArB12.iloc[:, 2:].loc[data_ArB12['group'] == 'A'].transpose()
df_heat_spec7 = data_ArB12.iloc[:, 2:].loc[data_ArB12['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(df_heat_spec6, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec7, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap="YlGnBu"

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('12h_ArB.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()


###### Heatmap 12b ############
suptitle2 = "Transcripts with 12h period in SCNx/tRF"

###### Heatmap plot ############

# BrA heat data

df_heat_spec8 = data_BrA12.iloc[:, 2:].loc[data_BrA12['group'] == 'A'].transpose()
df_heat_spec9 = data_BrA12.iloc[:, 2:].loc[data_BrA12['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(df_heat_spec8, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec9, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap) 

fig.suptitle(suptitle2, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.savefig('12h_BrA.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()

#####################################


#### CHECKING RAW COUNTS ####
# BUT they cannot be used due to big diff in number of reads in one RF sample 3


###### Heatmap counts24a ############
suptitle1 = "_counts24 Transcripts rhythmic in Sham/ad lib"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
# ArB heat data

df_heat_spec10 = data_ArB_counts24.iloc[:, 2:].loc[data_ArB_counts24['group'] == 'A'].transpose()
df_heat_spec11 = data_ArB_counts24.iloc[:, 2:].loc[data_ArB_counts24['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(df_heat_spec10, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec11, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap=cmap

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('ArB_counts24.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()


###### Heatmap counts24b ############
suptitle2 = "_counts24 Transcripts rhythmic in SCNx/tRF"

###### Heatmap plot ############

# BrA heat data

df_heat_spec12 = data_BrA_counts24.iloc[:, 2:].loc[data_BrA_counts24['group'] == 'A'].transpose()
df_heat_spec13 = data_BrA_counts24.iloc[:, 2:].loc[data_BrA_counts24['group'] == 'B'].transpose()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(df_heat_spec12, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=40, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(df_heat_spec13, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap) 

fig.suptitle(suptitle2, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.savefig('BrA_counts24.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()

"""

##########################################################################

"""
#######################################################################################################################################
######### Processed Data Loading - Transcripts rhythmic in both groups, sorted by phase in A or B group #############
#######################################################################################################################################

cmap='viridis_r'
#cmap='YlGnBu'

### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)
# causes problem with rotation of ylabel ? #

# only amplitudes > 1
data24_ArBr = pd.read_csv('biocyc24_ArBr.csv')
data_compl_A = data24_ArBr.loc[data24_ArBr['AMPLITUDE_A_Biocycle'] >= 1].sort_values(by = ['LAG_A_Biocycle'])[['Name','A_00_fpkm', 'A_03_fpkm', 'A_06_fpkm', 'A_09_fpkm', 'A_12_fpkm',
       'A_15_fpkm', 'A_18_fpkm', 'A_21_fpkm', 'B_00_fpkm', 'B_03_fpkm',
       'B_06_fpkm', 'B_09_fpkm', 'B_12_fpkm', 'B_15_fpkm', 'B_18_fpkm',
       'B_21_fpkm']].set_index('Name').apply(lambda x: x/x.max(), axis=1)
data_compl_B = data24_ArBr.loc[data24_ArBr['AMPLITUDE_B_Biocycle'] >= 1].sort_values(by = ['LAG_B_Biocycle'])[['Name','A_00_fpkm', 'A_03_fpkm', 'A_06_fpkm', 'A_09_fpkm', 'A_12_fpkm',
       'A_15_fpkm', 'A_18_fpkm', 'A_21_fpkm', 'B_00_fpkm', 'B_03_fpkm',
       'B_06_fpkm', 'B_09_fpkm', 'B_12_fpkm', 'B_15_fpkm', 'B_18_fpkm',
       'B_21_fpkm']].set_index('Name').apply(lambda x: x/x.max(), axis=1)
data1 = data_compl_A[['A_00_fpkm', 'A_03_fpkm', 'A_06_fpkm', 'A_09_fpkm', 'A_12_fpkm',
       'A_15_fpkm', 'A_18_fpkm', 'A_21_fpkm']]
data2 = data_compl_A[['B_00_fpkm', 'B_03_fpkm', 'B_06_fpkm', 'B_09_fpkm', 'B_12_fpkm', 'B_15_fpkm', 'B_18_fpkm',
       'B_21_fpkm']]
data3 = data_compl_B[['A_00_fpkm', 'A_03_fpkm', 'A_06_fpkm', 'A_09_fpkm', 'A_12_fpkm',
       'A_15_fpkm', 'A_18_fpkm', 'A_21_fpkm']]
data4 = data_compl_B[['B_00_fpkm', 'B_03_fpkm', 'B_06_fpkm', 'B_09_fpkm', 'B_12_fpkm', 'B_15_fpkm', 'B_18_fpkm',
       'B_21_fpkm']]


# all amplitudes
#data1 = pd.read_csv('biocyc24_ArBr_lagA_A.csv', index_col='Name')   # norm fpkm of genes rhythmic in A sorted by lag (phase) of genes rh. in A
#data2 = pd.read_csv('biocyc24_ArBr_lagA_B.csv', index_col='Name')   # etc.
#data3 = pd.read_csv('biocyc24_ArBr_lagB_A.csv', index_col='Name')
#data4 = pd.read_csv('biocyc24_ArBr_lagB_B.csv', index_col='Name')

##### prepare plot style ##########
sns.set_context("paper")   # , font_scale=1.2
sns.set_palette("Set1", 8, .75)
sns.set_style("ticks")  # "ticks", "white"

###### Heatmaps ############
suptitle1 = "Transcripts rhythmic in both groups"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(data1, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(data2, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap="YlGnBu"

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('HiAmp_ArBr_lagA.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(data3, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SCNx = sns.heatmap(data4, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap=cmap

fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('HiAmp_ArBr_lagB.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()
"""

#####################################
"""
####### Histograms of lag/phase ######
data = pd.read_csv('biocyc24_ArBr.csv')


######## Single Histogram ##########
y = "LAG_B_Biocycle"  
x_lab = 'Phase (Biocycle lag)'
y_lab = "Frequency"
ylim = (0, 0.1)
xlim = (0, 24)
suptitle_all = f'{x_lab} {y}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)
xticks = np.linspace(0,21,8)   # min, max, number of values for ticks on x axis

allplot = sns.FacetGrid(data)
allplot = (allplot.map(sns.distplot, y, bins=12)).set(xlim=xlim, xticks=xticks, xlabel=x_lab, ylabel=y_lab)
#plt.legend(title='Sex')
plt.text(x_coord, y_coord, f'n = ' + str(data[y].size - data[y].isnull().sum()) + '\nmean = ' + str(round(data[y].mean(), 3)) + ' ± ' + str(round(data[y].sem(), 3)) + 'h')  #data['Bamid'].mean() #misto centered

#plt.show()
plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
#plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')
plt.clf()
plt.close()
"""

"""
######## Regression + histo plots #########
y = "LAG_A_Biocycle"
x = "LAG_B_Biocycle"
y_lab = y
x_lab = x
xlim = (-3, 24)       #ylim = (data[y].min() - (data[y].max()-data[y].min())/8, data[y].max() + (data[y].max()-data[y].min())/8)
ylim = (-3, 24)        #xlim = (data[x].min() - (data[x].max()-data[x].min())/8, data[x].max() + (data[x].max()-data[x].min())/8)
suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)

allplot = sns.jointplot(x, y, data=data, kind='reg', xlim=xlim, ylim=ylim, scatter_kws={'s':8}, marginal_kws=dict(bins=24))    #scatter_kws={'color':'red'}
newindex = ~(np.isnan(data[x]) | np.isnan(data[y]))     #alt - just call pearson on data.x.dropnan()
r, p = stats.pearsonr(data[x][newindex], data[y][newindex])
plt.text(x_coord, y_coord, 'pearson r = ' + str(round(r, 3)) + ', P = ' + str(round(p, 6)) + ', n = ' + str(newindex.sum()))
plt.xlabel(x_lab)
plt.ylabel(y_lab)
allplot.fig.suptitle(suptitle_all, fontsize=14, fontweight='bold')

#plt.show()

#plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
#plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')
plt.clf()
plt.close()
"""



"""
######## Regression + histo plots of A vs B FPKM values od all time points #########
data_fpkm = pd.read_csv('fpkm_filtered.csv')
data=data_fpkm

# Prepare df with only A and B columns for all time points
data_a = pd.DataFrame()
for i in range(len(data.columns[1:9])):
    #print(data[['Ensembl', data.columns[i], data.columns[i+8]]])
    data[['Ensembl', 'A', 'B']] = data[['Ensembl', data.columns[i+1], data.columns[i+8]]]
    data_a = data_a.append(data[['Ensembl', 'A', 'B']])
    
# Filter out low and high FPKM values
data = data_a.loc[(data_a.A < 1000) & (data_a.A > 1) & (data_a.B < 1000) & (data_a.B > 1)]

# Plot
x = "A"
y = "B"
y_lab = y
x_lab = x
xlim = ylim = (data[y].min() - (data[y].max()-data[y].min())/8, data[y].max() + (data[y].max()-data[y].min())/8)
ylim = xlim = (data[x].min() - (data[x].max()-data[x].min())/8, data[x].max() + (data[x].max()-data[x].min())/8)
suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)

allplot = sns.jointplot(x, y, data=data, kind='reg', xlim=xlim, ylim=ylim, scatter_kws={'s':8}, marginal_kws=dict(bins=24))    #scatter_kws={'color':'red'}
newindex = ~(np.isnan(data[x]) | np.isnan(data[y]))     #alt - just call pearson on data.x.dropnan()
r, p = stats.pearsonr(data[x][newindex], data[y][newindex])
plt.text(x_coord, y_coord, 'pearson r = ' + str(round(r, 3)) + ', P = ' + str(round(p, 6)) + ', n = ' + str(newindex.sum()))
plt.xlabel(x_lab)
plt.ylabel(y_lab)
allplot.fig.suptitle(suptitle_all, fontsize=14, fontweight='bold')

#plt.show()

#plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
#plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')
plt.clf()
plt.close()

"""


"""
###############################################################################################
####### Polar Histogram of frequency of phases (LAG from Biocycle) ########
###############################################################################################

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

data_Ar = pd.read_csv('biocyc24_Ar_Pval.csv')
data_Br = pd.read_csv('biocyc24_Br_Pval.csv')

# Which data?
ab = 'A'
data = data_Ar.loc[data_Ar['AMPLITUDE'] >= 1]

# Use amplitude to filter out outliers or nans
#outlier_reindex = ~(np.isnan(reject_outliers(data[['Amplitude']])))['Amplitude']          # need series of bool values for indexing 
outlier_reindex = ~(np.isnan(data['Q_VALUE']))

data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

phaseseries = data_filt['LAG'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Q_VALUE'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['ID'].values.flatten()                   #.astype(int)                      # plot profile name as color

# LENGTH (AMPLITUDE)
amp = data_filt['AMPLITUDE'].values.flatten()                       # plot filtered Amplitude as length
#amp = 1                                                            # plot arbitrary number if Amp problematic

# POSITION (PHASE)
phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
#phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
#phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using CI or SEM of phase, which is in hours
phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 



N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
#colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html
colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap

phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=18)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=18)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')
axh.yaxis.grid(False)   # turns off circles
axh.xaxis.grid(False)  # turns off radial grids
axh.tick_params(pad=-2)   # moves labels closer or further away from subplots
#axh.set_xlabel(f'{title}', fontsize=2, labelpad=1)   # place specific title and use labelpad to adjust proximity to subplot


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
#axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black')) #add arrow

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'Histogram_Phase_{ab}_hiamp.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'Histogram_Phase_{ab}_hiamp.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()

"""



"""
###### Compare A and B group parameters with histograms ######
####### Select data to plot ########

data_Ar = pd.read_csv('biocyc24_Ar_Pval.csv')
data_Br = pd.read_csv('biocyc24_Br_Pval.csv')

data_Ar['GROUP'] = 'A'
data_Br['GROUP'] = 'B'

# Use Q value to filter out outliers or nans
outlier_reindex = ~(np.isnan(data_Ar['Q_VALUE']))
data_Ar = data_Ar[data_Ar.columns[:].tolist()][outlier_reindex]
outlier_reindex = ~(np.isnan(data_Br['Q_VALUE']))
data_Br = data_Br[data_Br.columns[:].tolist()][outlier_reindex]

data = data_Ar.merge(data_Br, how='outer')
data['LOG_AMP'] = np.log(data['AMPLITUDE'])  # log amplitude because large variation and not normal distribution

# to compare only genes rhythmic in both A and B
#data = data.loc[data['ID'].duplicated(keep=False) == True]   # return genes significant in both A and B

#col_var = 'LAG_cat'   #try to categorize by lag and then compare amplitude?
#col_dat = data.agecat_3
hue = "GROUP"
hue_dat = data.GROUP
hue_order = ['A', 'B']
y = "LOG_AMP"
x = y
x_lab = y
y_lab = "Frequency"
ylim = (0, 0.8)
xlim = (-4, 7)
#suptitle_all = f'{x_lab} vs {y_lab}'
suptitle_all = 'LOG Amplitude in same genes in A vs. B'
x_coord = xlim[0] + (xlim[1]/8)
y_coord = ylim[1] - (ylim[1]/8)


#g = sns.FacetGrid(data, col=col_var, col_order=col_order, hue=hue, sharex=True)
g = sns.FacetGrid(data, hue=hue, sharex=True)
g = (g.map(sns.distplot, y)).set(xlim=xlim).set_axis_labels(x_lab, y_lab)   #, bins=24

###### Calculate t test p values between hue_dat for separate categories in col_dat ######
pvalues = []
#for title in col_order:
    #datax1 = data[y][(hue_dat == hue_order[0]) & (col_dat == title)].dropna(how='any')
    #datax2 = data[y][(hue_dat == hue_order[1]) & (col_dat == title)].dropna(how='any')
datax1 = data[y][hue_dat == hue_order[0]].dropna(how='any')
datax2 = data[y][hue_dat == hue_order[1]].dropna(how='any')
t, p = stats.ttest_ind(datax1.values, datax2.values)
pvalues = pvalues + [p]  

######## Add calculated  p values to each subplot ##########
#for ax, title, p in zip(g.axes.flat, col_order, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
for ax, p in zip(g.axes.flat, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
    #ax.set_title(title, pad=-8)
    ax.text(3, 0.4, 't test \nP = ' + str(round(p, 5)), fontsize=10)

####### Labels, titles, axis limits, legend ################
ax.legend(title=None, loc='center right', fontsize='x-small')
#plt.xlim(xlim)
#plt.xlabel("Chronotype (h around midnight)")
#plt.ylabel("Frequency")
plt.suptitle(suptitle_all)

plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')


"""






#####################################


##### TEXT MANIPULATIONS ############

"""
##### Create individual sample tables from common table
dataA = pd.read_csv('EcountsB.csv')

for i in dataA.columns[1:]:
    df = pd.DataFrame(dataA[['Gene', f'{i}']])
    print(df.head())
    df.to_csv(f'{i}.csv', sep = '\t', index=False)  #create Tab-delimited csv 
"""

"""
##### Change format of csv to tabular for Galaxy
data = pd.read_csv('group_B_double.csv')
data.to_csv('t_group_B_double.csv', sep = '\t', index=False)
"""

"""
#### Remove duplicate entries
data.drop_duplicates('column_name') #drop in dupl in spec column
"""

"""
#### Change type of 1 column
data6.loc['ENTREZID'] = data6.ENTREZID.astype(int)
"""

"""
#### Rename columns
data8.columns = ['ENTREZID', 'Symbol']
"""

#####################################
"""
##### Match GeneID and ENSEMBL ID ############
data1 = pd.read_csv('deseq2_results.csv')
data2 = pd.read_csv('counts_to_match_IDs.csv')
data3 = pd.read_csv('fpkm_cycle_to_match_IDs.csv')  # couple gene incorrect name due to excel, need hand adjustments
data_counts = data2.merge(data1, how = 'outer')
print(data_counts.head())
data = data_counts.merge(data3, how = 'outer')
print(data.head())
data.to_csv('data_merged.csv', index=False)
"""
"""
#### Match GeneID and ENSEMBL ID for 12h period genes in both samples
names = pd.read_csv('Names_12h_both.csv')
data = data3.merge(names, how='outer')
data[['GeneID', 'DEG']].to_csv('both_12h_r_ids.csv', sep = '\t', index=False)
"""


# Lukac project
"""
clipped_mouse = pd.read_csv ('ClippedMouse.csv')
mouse = pd.DataFrame(clipped_mouse, columns=['Gene stable ID', 'Gene name', 'NCBI gene (formerly Entrezgene) ID']).drop_duplicates('Gene stable ID')
mouse = mouse.rename(columns={'Gene stable ID': 'ENSEMBL', 'Gene name': 'Symbol', 'NCBI gene (formerly Entrezgene) ID': 'EntrezGene ID'})
mouse = mouse.loc[~(mouse['EntrezGene ID'].isna())]
mouse = mouse.drop_duplicates('ENSEMBL')

rat0 = pd.read_csv('rat_gene_IDs.csv')
rat = pd.DataFrame(rat0, columns=['Gene stable ID', 'Gene name', 'NCBI gene (formerly Entrezgene) ID', 'RGD ID', 'Gene description']).drop_duplicates('Gene stable ID')
rat = rat.rename(columns={'Gene stable ID': 'ENSEMBL', 'Gene name': 'Symbol', 'NCBI gene (formerly Entrezgene) ID': 'EntrezGene ID', 'Gene description': 'Name'})
rat = rat.loc[~(rat['EntrezGene ID'].isna())]
rat = rat.drop_duplicates('ENSEMBL')
rat = rat.astype({'EntrezGene ID': 'int'})   # was float

homology0 = pd.read_csv('vertebrate_homology_classes.csv')
columns = ['HomoloGene ID', 'Common Organism Name', 'Symbol',
       'EntrezGene ID', 'Mouse MGI ID', 'Name', 'Synonyms']
data = pd.DataFrame(homology0, columns=columns)
homology = data.loc[(data['Common Organism Name'] == 'mouse, laboratory') | (data['Common Organism Name'] == 'rat')]

merge1 = homology.merge(mouse, on=['EntrezGene ID', 'Symbol'], how='outer')

merge2 = merge1.merge(rat, on='EntrezGene ID', how='outer')
df0 = merge2.loc[~(merge2['HomoloGene ID'].isna())]
df = df0.drop(columns=['Name_y', 'Symbol_y'])
df = df.rename(columns={'Symbol_x': 'Symbol', 'Name_x': 'Name', 'ENSEMBL_x': 'ENS_MOUSE', 'ENSEMBL_y': 'ENS_RAT'})

#uns = df.unstack('Common Organism Name')
#pivot1 = df.pivot(index='HomoloGene ID', columns='Common Organism Name', values=['Symbol', 'EntrezGene ID', 'Mouse MGI ID', 'Name', 'Synonyms', 'ENSEMBL'])

"""






# PROTEOMICS

#######################################################################################################################################
######### Processed Data Loading - Proteins rhythmic in A or B groups, sorted by phase in A or B group #############
#######################################################################################################################################

"""
# Raw data
data1 = pd.read_csv('ArA_lagA.csv', index_col='Name')   # raw log2 of proteins rhythmic in A sorted by lag (phase) of genes rh. in A
data2 = pd.read_csv('BrA_lagA.csv', index_col='Name')   # etc.
data3 = pd.read_csv('ArB_lagB.csv', index_col='Name')
data4 = pd.read_csv('BrB_lagB.csv', index_col='Name')
"""

"""
sns.set_context("paper")   # , font_scale=1.2
#sns.set_palette("husl")
sns.set_palette("Set1", 8, .75)
#sns.set_palette("Set3", 10)
sns.set_style("ticks")  # "ticks", "white"

cmap='viridis_r'
#cmap='YlGnBu'


### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)
# causes problem with rotation of ylabel ? #



data1 = pd.read_csv('normArA_lagA.csv', index_col='Name')   # norm log2 of proteins rhythmic in A sorted by lag (phase) of genes rh. in A
data2 = pd.read_csv('normBrA_lagA.csv', index_col='Name')   # etc.
data3 = pd.read_csv('normArB_lagB.csv', index_col='Name')
data4 = pd.read_csv('normBrB_lagB.csv', index_col='Name')


##### prepare plot style ##########
sns.set_context("paper")   # , font_scale=1.2
sns.set_palette("Set1", 8, .75)
sns.set_style("ticks")  # "ticks", "white"

###### Heatmaps ############
suptitle1 = "Proteins rhythmic in SHAM LD"
suptitle2 = "Proteins rhythmic in SCNx RF"
titleA = "Sham, ad lib"
titleB = "SCNx, tRF"
x_lab = "time"

###### Heatmap plot ############
fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
#heat_SHAM = sns.heatmap(data1, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21, 24], yticklabels=10, annot=False, cbar=False, ax=axs[0], cmap=cmap)  # need to tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label 
heat_SHAM = sns.heatmap(data1, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21, 24], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap) 
heat_SCNx = sns.heatmap(data2, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21, 24], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap=cmap

#fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('_norm_rhythmicA_lagA.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()


fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_SHAM = sns.heatmap(data3, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21, 24], yticklabels=False, annot=False, cbar=False, ax=axs[0], cmap=cmap)
heat_SCNx = sns.heatmap(data4, xticklabels=[0, 3, 6, 9, 12, 15, 18, 21, 24], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap=cmap)   # , cmap="YlGnBu"

#fig.suptitle(suptitle2, fontsize=14, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='CT (h)')
axs[1].set_title(titleB, fontsize=10, fontweight='bold')
axs[1].set(xlabel='CT (h)')

#plt.show()

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig('_norm_rhythmicB_lagB.svg', format = 'svg', bbox_inches = 'tight')
plt.clf()
plt.close()

"""


"""
###############################################################################################
####### Polar Histogram of frequency of phases (LAG from Biocycle) with Rayleigh vector########
###############################################################################################

def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r


data_Ar = pd.read_csv('Prot_biocyc24_Ar_Pval.csv')
data_Br = pd.read_csv('Prot_biocyc24_Br_Pval.csv')

# Which data?
ab = 'A'
data=data_Ar

# Use amplitude to filter out outliers or nans
#outlier_reindex = ~(np.isnan(reject_outliers(data[['Amplitude']])))['Amplitude']          # need series of bool values for indexing 
outlier_reindex = ~(np.isnan(data['Q_VALUE']))

data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

phaseseries = data_filt['LAG'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Q_VALUE'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['ID'].values.flatten()                   #.astype(int)                      # plot profile name as color

#colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))     # gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

# LENGTH (AMPLITUDE)
amp = data_filt['AMPLITUDE'].values.flatten()                       # plot filtered Amplitude as length
#amp = 1                                                            # plot arbitrary number if Amp problematic

# POSITION (PHASE)
phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
#phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
#phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using CI or SEM of phase, which is in hours
phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 



N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
#colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html
colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap


phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=18)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=18)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')
axh.yaxis.grid(False)   # turns off circles
axh.xaxis.grid(False)  # turns off radial grids
axh.tick_params(pad=-2)   # moves labels closer or further away from subplots



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
#axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black')) #add arrow

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'Histogram_Phase_{ab}.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'Histogram_Phase_{ab}.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()

"""


"""
###### Compare A and B group parameters with histograms ######
####### Select data to plot ########

data_Ar = pd.read_csv('Prot_biocyc24_Ar_Pval.csv')
data_Br = pd.read_csv('Prot_biocyc24_Br_Pval.csv')

data_Ar['GROUP'] = 'A'
data_Br['GROUP'] = 'B'

# Use Q value to filter out outliers or nans
outlier_reindex = ~(np.isnan(data_Ar['Q_VALUE']))
data_Ar = data_Ar[data_Ar.columns[:].tolist()][outlier_reindex]
outlier_reindex = ~(np.isnan(data_Br['Q_VALUE']))
data_Br = data_Br[data_Br.columns[:].tolist()][outlier_reindex]

data = data_Ar.merge(data_Br, how='outer')
data['LOG_AMP'] = np.log(data['AMPLITUDE'])  # log amplitude because large variation and not normal distribution

# to compare only genes rhythmic in both A and B
#data = data.loc[data['ID'].duplicated(keep=False) == True]   # return genes significant in both A and B

#col_var = 'LAG_cat'   #try to categorize by lag and then compare amplitude?
#col_dat = data.agecat_3
hue = "GROUP"
hue_dat = data.GROUP
hue_order = ['A', 'B']
y = "LOG_AMP"
x = y
x_lab = y
y_lab = "Frequency"
ylim = (0, 0.8)
xlim = (-4, 0.5)
#suptitle_all = f'{x_lab} vs {y_lab}'
suptitle_all = 'LOG Amplitude in A vs. B'
x_coord = xlim[0] + (xlim[1]/8)
y_coord = ylim[1] - (ylim[1]/8)


#g = sns.FacetGrid(data, col=col_var, col_order=col_order, hue=hue, sharex=True)
g = sns.FacetGrid(data, hue=hue, sharex=True)
g = (g.map(sns.distplot, y)).set(xlim=xlim).set_axis_labels(x_lab, y_lab)   #, bins=24

###### Calculate t test p values between hue_dat for separate categories in col_dat ######
pvalues = []
#for title in col_order:
    #datax1 = data[y][(hue_dat == hue_order[0]) & (col_dat == title)].dropna(how='any')
    #datax2 = data[y][(hue_dat == hue_order[1]) & (col_dat == title)].dropna(how='any')
datax1 = data[y][hue_dat == hue_order[0]].dropna(how='any')
datax2 = data[y][hue_dat == hue_order[1]].dropna(how='any')
t, p = stats.ttest_ind(datax1.values, datax2.values)
pvalues = pvalues + [p]  

######## Add calculated  p values to each subplot ##########
#for ax, title, p in zip(g.axes.flat, col_order, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
for ax, p in zip(g.axes.flat, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
    #ax.set_title(title, pad=-8)
    ax.text(-1, 1, 't test \nP = ' + str(round(p, 5)), fontsize=10)

####### Labels, titles, axis limits, legend ################
ax.legend(title=None, loc='center right', fontsize='x-small')
#plt.xlim(xlim)
#plt.xlabel("Chronotype (h around midnight)")
#plt.ylabel("Frequency")
plt.suptitle(suptitle_all)

plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')
"""
