import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt

def checkResult(Wa,Wb,Wc,prints,limit):
    ret=0
    p=Wa.shape[0]
    nn=Wa.shape[1]
    n=int(np.sqrt(nn))

    cStar=np.zeros([p,nn**2])
    cWave=np.zeros([nn,nn**2])

    for i in range(p):
        cStar[i,:]=[Wa[i,kk]*Wb[i,k] for kk in range(nn) for k in range(nn)]

    cWave=Wc.dot(cStar)

    cStarTxt=["a("+str(int((kk-kk%n)/n))+","+str(int(kk%n))+")"
    +"*b("+str(int((k-k%n)/n))+","+str(int(k%n))+")"
    for kk in range(nn) for k in range(nn)]

    cWaveTxt=np.chararray((n,n),unicode=True,itemsize=256)
    cTxt=np.chararray((n,n),unicode=True,itemsize=256)

    for i in range(n):
        for j in range(n):
            cWaveTxt[i,j]=''
            for k in range(nn**2):
                if abs(cWave[i*n+j,k]-1.0)<0.5:
                    cWaveTxt[i,j]+=cStarTxt[k]+"+"
                elif abs(cWave[i*n+j,k])>0.5:
                    cWaveTxt[i,j]+=str('%2.4f'%cWave[i*n+j,k])+"*"+cStarTxt[k]+"+"
            cWaveTxt[i,j]=cWaveTxt[i,j][0:max(len(cWaveTxt[i,j])-1,0)]

    for i in range(n):
        for j in range(n):
            cTxt[i,j]=''
            for k in range(n):
                cTxt[i,j]+="a("+str(i)+","+str(k)+")*b("+str(k)+","+str(j)+")+"
            cTxt[i,j]=cTxt[i,j][0:max(len(cTxt[i,j])-1,0)]

    for i in range(n):
        for j in range(n):
            if not cTxt[i,j]==cWaveTxt[i,j]:
                ret+=1
                if prints:
                    print("Fehler in C("+str(i)+","+str(j)+"):")
                    print("C_correct = "+cTxt[i,j])
                    print("C_learned = "+cWaveTxt[i,j])
    return ret

# Wa,Wb,Wc=np.load("WaWbWc.npy")
# checkResult(Wa,Wb,Wc,True)
