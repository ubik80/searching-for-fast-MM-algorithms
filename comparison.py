import numpy as np
import math as mt
np.set_printoptions(precision=2, suppress=True)

W1=np.load("bestSolution_backup_n_3_100Proz_15Mai.npy",allow_pickle=True)
W2=np.load("bestSolution_backup_n_3_100Proz_17Mai.npy",allow_pickle=True)
W3=np.load("bestSolution_backup_n_3_100Proz_18Mai.npy",allow_pickle=True)
W4=np.load("bestSolution_backup_n_3_100Proz_18Mai_2.npy",allow_pickle=True)
W5=np.load("solution_3_1_DiffMap_28Mai.npy",allow_pickle=True)
W6=np.load("solution_3_2_DiffMap_30Mai_1.npy",allow_pickle=True)
W7=np.load("solution_3_3_DiffMap_30Mai_2.npy",allow_pickle=True)
W8=np.load("solution_3_7_DiffMap_25Mai.npy",allow_pickle=True)

W1=W1[0:3]
W2=W2[0:3]
W3=W3[0:3]
W4=W4[0:3]
W5=W5[0:3]
W6=W6[0:3]
W7=W7[0:3]
W8=W8[0:3]

nn=W1[0].shape[1]
p=W1[0].shape[0]
n=int(np.round(np.sqrt(nn)))

Ws=np.row_stack([W1,W2,W3,W4,W5,W6,W7,W8])
Rs=np.zeros([8,3])

q=0
for W in Ws:
    ranks=np.zeros(3)
    for i in range(p):
        M=W[0][i,:].reshape([n,n])
        r=int(np.linalg.matrix_rank(M))
        ranks[r-1]+=1
    for i in range(p):
        M=W[1][i,:].reshape([n,n])
        r=int(np.linalg.matrix_rank(M))
        ranks[r-1]+=1
    for i in range(p):
        M=W[2][:,i].reshape([n,n])
        r=int(np.linalg.matrix_rank(M))
        ranks[r-1]+=1
    Rs[q,:]=ranks
    q+=1

print(Rs)

W1[1]
W6[1]

for i in range(p):
    for ii in range(p):
        if np.array_equal(W1[1][i,:],W6[1][ii,:]): print(i,ii)

for j in range(nn):
    for jj in range(nn):
        if np.array_equal(W1[1][:,jj],W6[1][:,jj]): print(j,jj)

for i in range(nn):
    for ii in range(nn):
        if np.array_equal(W1[1][:,i],W6[1][:,ii]): print(i,ii)

WW=W6

sum=0
W=WW[0]
for i in range(p):
    sum+=max(0,np.sum(W[i,:]!=0)-1)
W=WW[1]
for i in range(p):
    sum+=max(0,np.sum(W[i,:]!=0)-1)
W=WW[2]
for j in range(p):
    sum+=max(0,np.sum(W[:,j]!=0)-1)

sum
