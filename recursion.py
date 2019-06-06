import numpy as np

W=np.load("bestSolution_backup_n_3_100Proz_15Mai.npy",allow_pickle=True)

nn=W[0].shape[1]
p=W[0].shape[0]
n=int(np.round(np.sqrt(nn)))

r=30
rr=int(r*r)

Wa=W[0]
Wb=W[1]
Wc=W[2]

A=np.random.rand(n*r,n*r)
B=np.random.rand(n*r,n*r)
C=A.dot(B)

AA=[]
for i in range(n):
    aa=np.zeros([r,r])
    for j in range(n):
        AA.append(A[i*r:(i+1)*r,j*r:(j+1)*r])
BB=[]
for i in range(n):
    for j in range(n):
        BB.append(B[i*r:(i+1)*r,j*r:(j+1)*r])

CStar=[]
for i in range(p):
    aStar=np.zeros([r,r])
    bStar=np.zeros([r,r])
    for j in range(nn):
        aStar+=Wa[i,j]*AA[j]
        bStar+=Wb[i,j]*BB[j]
    cStar=aStar.dot(bStar)
    CStar.append(cStar)

CC=[]
for i in range(nn):
    cc=np.zeros([r,r])
    for j in range(p):
        cc+=Wc[i,j]*CStar[j]
    CC.append(cc)

CWave=np.zeros(C.shape)
for p in range(n):
    for q in range(n):
        c=CC[p*n+q]
        for i in range(r):
            for j in range(r):
                CWave[p*r+i,q*r+j]=c[i,j]

deviation=False
for i in range(r*n):
    for j in range(r*n):
        if abs(CWave[i,j]-C[i,j])>0.0000001: deviation=True

deviation



#
