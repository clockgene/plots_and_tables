import pandas as pd
import os


from tkinter import filedialog
from tkinter import *

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
df = df.astype({'HomoloGene ID': 'int'})

print(df[df.duplicated() == True])
df1 = df.drop_duplicates()   #removes just 1 row

# testing pivot table
dfp1 = df1.iloc[:100, :].pivot(index='HomoloGene ID', columns='Common Organism Name')  #works, but larger slice raises duplicates error

# Try to deal with duplicities
def dupl(df1):    

    values = df1.loc[(df1.iloc[:, :2].duplicated() == True), 'HomoloGene ID'].values    
    counter = df1.index.max()
    for i in range(len(values)):
        values[i] = i + counter
        counter += 1

    return values

values = dupl(df1)
df1.loc[(df1.iloc[:, :2].duplicated() == True), 'HomoloGene ID'] = values       # raises warning, ident. values spread to multiple rows?

#dfpivot1 = df1.pivot(index='HomoloGene ID', columns='Common Organism Name')    #still 393 duplicated rows, so it does not work

# filtering out remaining duplicities
df_filt = df1[df1.iloc[:, :2].duplicated() == False]   #removes rows that are duplicated (e.g. rat has 2 copies of a single mouse gene)
dfpivot = df_filt.pivot(index='HomoloGene ID', columns='Common Organism Name')   

dfpivot.to_csv('dfpivot.csv')
duplicites = df1[df1.iloc[:, :2].duplicated() == True]
duplicites.to_csv('duplicites.csv')
