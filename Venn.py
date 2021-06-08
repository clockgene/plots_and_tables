import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib as mpl





### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)



# Use the venn2 function
venn2(subsets = (980, 1197, 679), set_labels = ('Group A', 'Group B'))
#plt.show()


plt.savefig(f'Venn.svg', format = 'svg')



