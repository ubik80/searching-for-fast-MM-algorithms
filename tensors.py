import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import math
np.set_printoptions(precision=2, suppress=True)

os.chdir("/Users/tillspaeth/Google Drive/V15DiffMap")
fileNames=os.listdir()
TfileNames=[]
for f in fileNames:
    if ".npy" in f: TfileNames.append(f)
fileNames=TfileNames

for f in fileNames:
    sol=np.load(f,allow_pickle=True)
    Wa=sol[0]
    Wb=sol[1]
    Wc=sol[2]
    nn=Wa.shape[1]
    n=int(math.sqrt(nn))
    p=Wa.shape[0]
    T=np.zeros([nn,nn,nn])
    for pp in range(r):
        
