import murmurhash as mh
import math as mt
import numpy as np

class bloomFilter(object):
    def __init__(self,n,P):
        self.m=int(-n*mt.log(P,2)/(mt.log(2,2)**2))
        self.K=int(self.m/n*mt.log(2,2))
        self.mem=np.zeros(self.m,dtype=bool)
        # print("Bloom Filter:")
        # print("P = ",P)
        # print("m = ",self.m)
        # print("k = ",self.K)
        return

    def mmhash(self,W,k):  
        p=W[0].shape[0]
        hs=np.zeros(p)
        WW=[W[0].copy(),W[1].copy()]
        for i in range(p):
            v=np.concatenate([WW[0][i,:],WW[1][i,:]])
            hs[i]=mh.hash(v.tobytes(),1)%1000000
        sorted=np.argsort(hs)
        WW[0]=WW[0][sorted,:]
        WW[1]=WW[1][sorted,:]
        h=mh.hash(np.concatenate([WW[0].flatten(),WW[1].flatten()]).tobytes(),k)
        return h%self.m

    def store(self,W):
        ret=True
        for k in range(self.K):
            h=self.mmhash(W,k)
            if self.mem[h]==False: ret=False
            self.mem[h]=True
        return ret

if __name__ == '__main__':
    W=np.load("bestSolution_backup_n_3_100Proz_15Mai.npy",allow_pickle=True)
    nn=W[0].shape[1]
    n=int(mt.sqrt(nn))
    p=W[0].shape[0]

    BF=bloomFilter(3*nn*p,0.0001)

    BF.store(W)

#
