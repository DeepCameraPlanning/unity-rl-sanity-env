import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import math as mt
import time

import sys
import glob
import os
from pathlib import Path

def showMat(mat, figName="", isGrid=False, isSave=False, saveDir=""):
  if figName:
    plt.figure(figName,figsize=(5,4))
  else:
    plt.figure()
  plt.imshow(mat)
  plt.colorbar()
  plt.yticks(np.arange(0, float(np.shape(mat)[0])+1, 20.0),rotation='vertical')
  plt.yticks(fontsize=8)
  plt.xticks(np.arange(0, float(np.shape(mat)[1])+1, 20.0))
  plt.xticks(fontsize=8)
  plt.ylabel("x")
  plt.xlabel("y")
  if isGrid:
    plt.grid(color='r', linestyle='-', linewidth=1)
  # plt.savefig(saveDir+'/'+figName+".jpg")
  # plt.clf()

def main(args):

    lenArg = len(args)
    print(lenArg," and ",args) 

    lcovar = []
    ctr=0
    for arg in args:
      numpy_array = np.loadtxt(arg, delimiter=",")

      fig = go.Figure(data=[go.Scatter3d(x=numpy_array[:,0], y=numpy_array[:,1], z=numpy_array[:,2], mode='markers')])
      fig.show()

    return
if __name__ == "__main__":
    main(sys.argv[1:])