import os
import pickle
import numpy as np  # (pip install numpy)
from tkinter import Tk, filedialog


basePath = "/home/mbraun/NewIS"
colabBasePath = "/content/drive/My Drive/NewIS"
root = Tk()
root.withdraw()
annotationDictFileName = filedialog.askopenfilename(initialdir=basePath, filetypes=[("Text Dict","*.txt")])

if not annotationDictFileName:
    quit()

# loading a saved dict back in
with open(annotationDictFileName, 'rb') as handle:
   readDictList = pickle.loads(handle.read())

for segDict in readDictList:
    segDict['file_name'] = segDict['file_name'].replace(basePath, colabBasePath)
    segDict['image_id'] = segDict['image_id'].replace(basePath, colabBasePath)

# newAnnotationDictFileName = 'annotations_dict_bitmask_Train_colab.txt'
newAnnotationDictFileName = 'annotations_dict_bitmask_Validation_colab.txt'

with open(newAnnotationDictFileName, 'wb') as handle:
    pickle.dump(readDictList, handle)