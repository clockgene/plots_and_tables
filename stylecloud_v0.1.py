import stylecloud as sc
import io

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import glob, os, shutil
from tkinter import filedialog
from tkinter import *
import re

##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button


##################### https://stackoverflow.com/questions/34837707/how-to-extract-text-from-a-pdf-file
def convert_pdf_to_txt(path):
    '''Convert pdf content from a file path to text

    :path the file path
    '''
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()

    with io.StringIO() as retstr:
        with TextConverter(rsrcmgr, retstr, codec=codec,
                           laparams=laparams) as device:
            with open(path, 'rb') as fp:
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                password = ""
                maxpages = 0
                caching = True
                pagenos = set()

                for page in PDFPage.get_pages(fp,
                                              pagenos,
                                              maxpages=maxpages,
                                              password=password,
                                              caching=caching,
                                              check_extractable=True):
                    interpreter.process_page(page)

                return retstr.getvalue()

"""
if __name__ == "__main__":
    print(convert_pdf_to_txt('.\\Sladek-Neuropharmacology 2021.pdf'))

"""


# Specify FOLDER with PDFs
root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse to folder", command=browse_button)
buttonBrowse.grid()
mainloop()
path = os.getcwd() + '\\'


filescount = []
for root, dirs, files in os.walk(".", topdown=False):
    filescount.append(len(files))

# run function to convert all pdfs to list of strings
alltxt = []
for file in files:
    alltxt.append(convert_pdf_to_txt(f'.\\{file}'))

# create output file
pdffile = open('paper_all.txt', 'w', encoding="utf-8")

# write to file
for f in alltxt:
    pdffile.write(''.join(f))

# save as txt file
pdffile.close()

# filter
my_long_list = ['et al', 'fig', 'and', 'of', 'the', 'to', 'in', 'were', 'was',
                'that', 'a', 'on', 'after', 'from', 'their', 'but', 'or', 'be', 'are', 'was',
                'is', 'for', 'by', 'with', 'it', 'not', 'et', 'al', 'as', 'we', 'between', 'before', 'cid', 'also', 'h', 'j', 'an', 'at', 'i.e.',
                'p', 'did', 'this', 'i e', 'due', 'thus', 'who', 'which', 'f f', 'b', 'c', 'k', 'g', 'd', 'e', 'f', 'i', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'these', 'those', 'its', 'pio', 'because', 'they', 'https', 'org', 'than',
                'however', 'may', 'when', 'well', 'therefore', 'can', 'during', 'only', 'shown', 'used', 'have', 'shown', 'all', 'both',
                'while', 'there', 'each', 'based', 'could', 'whether', 'might', 'had', 'has', 'whle', 'found', 'such', 'via', 'using', 'been',
                'other', 'way']

# open txt file and generate word cloud png, specify icon shape 
sc.gen_stylecloud(file_path='paper_all.txt', custom_stopwords=my_long_list, icon_name='fas fa-clock', output_name='paper_all.png')

