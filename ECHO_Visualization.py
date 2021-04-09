### This script allows you to merge and visualize all kinds of dataframes!

import pandas as pd
from matplotlib import pyplot as plt

#Upload your ECHO input here
pre_df1 = pd.read_csv (r'C:\\...')
#Upload your ECHO output here
pre_df2 = pd.read_csv (r'C:\\...')

#Here we merge the file depending on our preference, default is inner which merges only genes found in both files
#Merging crash course: https://stackoverflow.com/questions/53645882/pandas-merging-101
df = pd.merge(pre_df1, pre_df2, left_on= 'Gene', right_on = 'Gene Name', how='inner')

#Here we drop the unnecessary columns from the ECHO output. Remember, Python starts counting from 0!
#We may also print the df to ensure we didn't cut off more than we wanted :)
df.drop(df.iloc[:, 9:25], inplace = True, axis = 1) 
print(df)

#Optional, save the current df for later use, otherwise comment it out
#df.to_csv(r'C:\\...', index = False)

### Now for the visualization part

#Set your time period here
x = (0, 3, 6, 9, 12, 15, 18, 21)

for i in range(0,len(df)):
    #The values in (df['n'][i]) have to correspond with x values
    y = (df['0'][i], df['3'][i], df['6'][i], df['9'][i], df['12'][i], df['15'][i], df['18'][i], df['21'][i])
    #Adjust the color and marker as per your preference
    plt.plot(x, y, color = 'blue', marker = 'o')
    plt.xlabel('Time')
    plt.xticks(x)
    #Change the string 'Gene' to correspond with your index
    plt.title(df['Gene'][i])
    plt.ylabel('Relative Expression')
    #Choose your directory to save files here
    plt.savefig(r'C:\\...\ ' + str(df['Gene'][i]) + '.png')
    plt.clf()