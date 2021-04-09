### This loop allows dataframe visualization of all kinds of circadian data!

import pandas as pd 
from matplotlib import pyplot as plt
#Upload your file here
df = pd.read_csv(r'C:\\***.csv')
#Optional, you may print your df to check if it loaded correctly
print(df)

#Set your time period here
x = (0, 3, 6, 9, 12, 15, 18, 21)

for i in range(0,len(df)):
    #The values in (df['n'][i]) have to correspond with x values
    y = (df['0'][i], df['3'][i], df['6'][i], df['9'][i], df['12'][i], df['15'][i], df['18'][i], df['21'][i])
    #Adjust the color and marker as per your preference
    plt.plot(x, y, color = 'blue', marker = 'o')
    plt.xlabel('Time')
    plt.xticks(x)
    #Change the string "Gene" to correspond with your index
    plt.title(df['Gene'][i])
    plt.ylabel('Relative Expression')
    #Choose your directory to save files here
    plt.savefig(r'C:\\***\ ' + str(df['Gene'][i]) + '.png')
    plt.clf()