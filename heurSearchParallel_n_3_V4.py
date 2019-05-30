# coding: utf8
import os
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import smallStuff as sm
import backprobIterationNUMBAV4 as bi
import multiprocessing as mp
import time
import uuid
np.set_printoptions(precision=2, suppress=True)

@jit(nopython=True,nogil=True,cache=True)
def rankWeight(i,j,Wa,Wb,Wc,baseDev,matSel=0):
    nn=Wa.shape[1]
    p=Wa.shape[0]
    n=int(np.sqrt(nn))
    if matSel==0:
        delta=np.round(Wa[i,j])-Wa[i,j]
        delta*=np.sum(Wb[i,:])
        deltaVec=Wc[:,i]*delta
    elif matSel==1:
        delta=np.round(Wb[i,j])-Wb[i,j]
        delta*=np.sum(Wa[i,:])
        deltaVec=Wc[:,i]*delta
    else:
        delta=np.round(Wc[i,j])-Wc[i,j]
        delta*=np.sum(Wa[j,:])*np.sum(Wb[j,:])
        deltaVec=baseDev*0
        deltaVec[i]+=delta
    ret=np.linalg.norm(deltaVec+baseDev,2)
    return ret

@jit(nopython=True,nogil=True,cache=True)
def findWeight(Wa,Wb,Wc,MA,MB,MC):
    nn=Wa.shape[1]
    p=Wa.shape[0]
    n=int(np.sqrt(nn))
    bestI=-999
    bestJ=-999
    bestErr=999999
    matSel=-9
    a=np.ones(nn)
    b=np.ones(nn)
    baseDev=Wc.dot(Wa.dot(a)*Wb.dot(b))-np.ones(nn)*n
    for i in range(p):
        for j in range(nn):
            if MA[i,j]==1:
                err=rankWeight(i,j,Wa,Wb,Wc,baseDev,0)
                if err<bestErr:
                    bestErr=err
                    bestI=i
                    bestJ=j
                    matSel=0
    for i in range(p):
        for j in range(nn):
            if MB[i,j]==1:
                err=rankWeight(i,j,Wa,Wb,Wc,baseDev,1)
                if err<bestErr:
                    bestErr=err
                    bestI=i
                    bestJ=j
                    matSel=1
    for i in range(nn):
        for j in range(p):
            if MC[i,j]==1:
                err=rankWeight(i,j,Wa,Wb,Wc,baseDev,2)
                if err<bestErr:
                    bestErr=err
                    bestI=i
                    bestJ=j
                    matSel=2
    return bestI,bestJ,bestErr,matSel

@jit(nopython=True,nogil=True,cache=True)
def resets(MA,MB,MC):
    nn=MA.shape[1]
    p=MA.shape[0]
    resets=np.random.randint(np.floor((3*p*nn-np.sum(MA)-np.sum(MB)-np.sum(MC))-1))+1
    for r in range(resets):
        ss=False
        while not ss:
            ii=np.random.randint(p)
            jj=np.random.randint(3*nn)
            if jj<nn:
                if MA[ii,jj]==0.0:
                    MA[ii,jj]=1.0
                    ss=True
            elif jj<2*nn:
                jj-=nn
                if MB[ii,jj]==0.0:
                    MB[ii,jj]=1.0
                    ss=True
            else:
                jj-=2*nn
                if MC[jj,ii]==0.0:
                    MC[jj,ii]=1.0
                    ss=True
    return

def intSolutionSearch(n,p,maxTries,maxNumIters,tol,
bestWa,bestWb,bestWc,bestMA,bestMB,bestMC,mutex,id):

    useFileSystem=False
    iterFact=1
    seed=int(time.time())+int(uuid.uuid4())+id
    seed=seed%135790
    np.random.seed(seed)
    n=int(n)
    p=int(p)
    nn=int(n**2)

    mutex.acquire()
    bestWa_np=np.frombuffer(bestWa,dtype='d').reshape([p,nn])
    bestWb_np=np.frombuffer(bestWb,dtype='d').reshape([p,nn])
    bestWc_np=np.frombuffer(bestWc,dtype='d').reshape([nn,p])
    bestMA_np=np.frombuffer(bestMA,dtype='d').reshape([p,nn])
    bestMB_np=np.frombuffer(bestMB,dtype='d').reshape([p,nn])
    bestMC_np=np.frombuffer(bestMC,dtype='d').reshape([nn,p])
    MA=bestMA_np.copy()
    MB=bestMB_np.copy()
    MC=bestMC_np.copy()
    Wa=bestWa_np.copy()
    Wb=bestWb_np.copy()
    Wc=bestWc_np.copy()
    print("Prozess ",id," initialisiert")
    oldBest=np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
    mutex.release()

    for tries in range(maxTries):

        i,j,err,matSel=findWeight(Wa,Wb,Wc,MA,MB,MC)
        if matSel==0:
            MA[i,j]=0.0
            Wa[i,j]=float(min(max(round(Wa[i,j]),-1.0),1.0))
        elif matSel==1:
            MB[i,j]=0.0
            Wb[i,j]=float(min(max(round(Wb[i,j]),-1.0),1.0))
        else:
            MC[i,j]=0.0
            Wc[i,j]=float(min(max(round(Wc[i,j]),-1.0),1.0))

        Wa,Wb,Wc,errHist,success=bi.findCalcRule(n,p,maxNumIters*iterFact,
        Wa,Wb,Wc,MA,MB,MC,tol,nue=0.1)
        iterFact=1

        if not success:
            resets(MA,MB,MC)
            iterFact=4

        mutex.acquire()
        print("-------------------------------------")
        print("Prozess: ",str(id))
        print("Durchgang: ",str(tries))
        print("Position: ",str([i,j])," Wa/Wb/Wc: ",str(matSel))

        if useFileSystem:
            WaF,WbF,WcF,MAF,MBF,MCF=np.load("bestSolution_n_3.npy",allow_pickle=True)
            np.copyto(bestWa_np,WaF)
            np.copyto(bestWb_np,WbF)
            np.copyto(bestWc_np,WcF)
            np.copyto(bestMA_np,MAF)
            np.copyto(bestMB_np,MBF)
            np.copyto(bestMC_np,MCF)
        numOfMltpls=np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
        if numOfMltpls<oldBest:
            oldBest=numOfMltpls
            print("Innovation - Daten tauschen")
            Wa=bestWa_np.copy()
            Wb=bestWb_np.copy()
            Wc=bestWc_np.copy()
            MA=bestMA_np.copy()
            MB=bestMB_np.copy()
            MC=bestMC_np.copy()
        elif not success:
            print("keine Lösung gefunden")
        else: #success
            print("Lösung gefunden")
            if numOfMltpls>np.sum(MA)+np.sum(MB)+np.sum(MC):
                print("beste Lösung bisher")
                np.copyto(bestWa_np,Wa)
                np.copyto(bestWb_np,Wb)
                np.copyto(bestWc_np,Wc)
                np.copyto(bestMA_np,MA)
                np.copyto(bestMB_np,MB)
                np.copyto(bestMC_np,MC)
                numOfMltpls=np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
                oldBest=numOfMltpls
                if numOfMltpls<150 or numOfMltpls%10==0:
                    np.save("bestSolution_n_3",[Wa,Wb,Wc,MA,MB,MC])
                    print("in Datei geschrieben")

        if np.sum(np.isnan(Wa))>0 or np.sum(np.isnan(Wb))>0 or np.sum(np.isnan(Wc))>0:
            print("NAN NAN NAN NAN NAN NAN NAN NAN NAN")
            if id==0: print("Daten tauschen")
            Wa=bestWa_np.copy()
            Wb=bestWb_np.copy()
            Wc=bestWc_np.copy()
            MA=bestMA_np.copy()
            MB=bestMB_np.copy()
            MC=bestMC_np.copy()

        sm.printMM(MA,MB,-999,-999,-999,-999)
        print('')
        sm.printM(MC,i,j)
        print(str(int(np.sum(MA)+np.sum(MB)+np.sum(MC)))+" ("+str(int(numOfMltpls))+")")

        if (numOfMltpls<=1):
            print("Untergrenze erreicht - ENDE")
            mutex.release()
            return
        mutex.release()
    return


if __name__ == '__main__':
    start = time.time()
    numOfProc=int(mp.cpu_count())
    print("Anzahl Prozessoren: ",numOfProc)

    n=3
    p=23
    nn=int(n**2)

    tol=0.01

    # mMA=np.ones([p,nn],dtype=float)
    # mMB=np.ones([p,nn],dtype=float)
    # mMC=np.ones([nn,p],dtype=float)
    # mWa=np.zeros([p,nn],dtype=float)
    # mWb=np.zeros([p,nn],dtype=float)
    # mWc=np.zeros([nn,p],dtype=float)
    # success=False
    # while not success:
    #     print("initiiere Lösung ...")
    #     mWa=np.random.rand(p,nn)*2.0-1.0
    #     mWb=np.random.rand(p,nn)*2.0-1.0
    #     mWc=np.random.rand(nn,p)*2.0-1.0
    #     mWa,mWb,mWc,errHist,success=bi.findCalcRule(n,p,20000000,
    #     mWa,mWb,mWc,mMA,mMB,mMC,tol,nue=0.1)
    # print("Lösung initialisiert")
    # np.save("bestSolution_n_3",[mWa,mWb,mWc,mMA,mMB,mMC])

    mWa,mWb,mWc,mMA,mMB,mMC=np.load("bestSolution_n_3.npy",allow_pickle=True)

    mutex=mp.Lock()
    bestWa=mp.RawArray('d',mWa.flatten())
    bestWb=mp.RawArray('d',mWb.flatten())
    bestWc=mp.RawArray('d',mWc.flatten())
    bestMA=mp.RawArray('d',mMA.flatten())
    bestMB=mp.RawArray('d',mMB.flatten())
    bestMC=mp.RawArray('d',mMC.flatten())

    procs=[mp.Process(target=intSolutionSearch,
    args=(n,p,50000,1000000,tol,bestWa,bestWb,bestWc,bestMA,bestMB,bestMC,mutex,i))
    for i in range(numOfProc)]

    for pp in procs:pp.start()
    for pp in procs:pp.join()

    bestWa_np=np.frombuffer(bestWa,dtype='d').reshape([p,nn])
    bestWb_np=np.frombuffer(bestWb,dtype='d').reshape([p,nn])
    bestWc_np=np.frombuffer(bestWc,dtype='d').reshape([nn,p])
    mWa=bestWa_np.copy()
    mWb=bestWb_np.copy()
    mWc=bestWc_np.copy()

    end = time.time()
    print("Dauer:", end - start)
    print("Wa:")
    print(mWa)
    print("Wb:")
    print(mWb)
    print("Wc:")
    print(mWc)
    print("Beispiel:")
    A=np.random.randint(-9,9,[n,n])
    B=np.random.randint(-9,9,[n,n])
    print("A:")
    print(A)
    print("B:")
    print(B)
    print("C=A*B exakt:")
    print(A.dot(B))
    a=np.array(A.reshape(n**2),dtype=float)
    b=np.array(B.reshape(n**2),dtype=float)
    c=mWc.dot(mWa.dot(a)*mWb.dot(b))
    C=c.reshape((n,n))
    print("C=Wc*(Wa*b o Wb*b):")
    print(C)







    #
