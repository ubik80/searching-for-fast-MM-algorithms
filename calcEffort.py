import numpy as np
np.set_printoptions(precision=2, suppress=True)

Wa,Wb,Wc=np.load("solution_3_1.npy",allow_pickle=True)
#Wa,Wb,Wc=np.load("temp.npy")

#Strassen:
# Wa=[[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
# Wa=np.array(Wa)
# Wb=[[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
# Wb=np.array(Wb)
# Wc=[[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,1,0,0,0],[1,-1,1,0,0,1,0]]
# Wc=np.array(Wc)

limit=0.0001
nn=Wc.shape[0]
p=Wc.shape[1]
n=int(np.round(np.sqrt(nn)))

# Wa=np.round(Wa)
# Wb=np.round(Wb)
# Wc=np.round(Wc)

print("Wa:")
print(Wa)
print("Wb:")
print(Wb)
print("Wc:")
print(Wc)

A=np.random.randint(-9,9,[n,n])
B=np.random.randint(-9,9,[n,n])
print("Beispiel:")
print("A:")
print(A)
print("B:")
print(B)
print("C=A*B exakt:")
print(A.dot(B))
a=np.array(A.reshape(n**2),dtype=float)
b=np.array(B.reshape(n**2),dtype=float)
c=Wc.dot(Wa.dot(a)*Wb.dot(b))
C=c.reshape((n,n))
print("C=Wc*(Wa*b o Wb*b):")
print(C)

#Wa,Wb,Wc,MA,MB=np.load("bestSolution_backup_n3_11Mai.npy",allow_pickle=True)



# numOfZerosWc=np.argwhere(np.abs(Wc)<limit).shape[0]
# numOfOnesWc=np.argwhere(np.abs(Wc-1)<limit).shape[0]
# numOfMinOnesWc=np.argwhere(np.abs(Wc+1)<limit).shape[0]
# numNonMultiplWc=numOfZerosWc+numOfOnesWc+numOfMinOnesWc
# numMulWc=nn*p-numNonMultiplWc
#
# numOfZerosWa=np.argwhere(np.abs(Wa)<limit).shape[0]
# numOfOnesWa=np.argwhere(np.abs(Wa-1)<limit).shape[0]
# numOfMinOnesWa=np.argwhere(np.abs(Wa+1)<limit).shape[0]
# numNonMultiplWa=numOfZerosWa+numOfOnesWa+numOfMinOnesWa
# numMulWa=p*nn-numNonMultiplWa
#
# numOfZerosWb=np.argwhere(np.abs(Wb)<limit).shape[0]
# numOfOnesWb=np.argwhere(np.abs(Wb-1)<limit).shape[0]
# numOfMinOnesWb=np.argwhere(np.abs(Wb+1)<limit).shape[0]
# numNonMultiplWb=numOfZerosWb+numOfOnesWb+numOfMinOnesWb
# numMulWb=p*nn-numNonMultiplWb
#
# #Annahmen:
# n=200000
# nn=int(n**2)
# p=int(np.ceil(n**2.8))
# numMulWa=p*(nn/p)
# numMulWb=p*(nn/p)
# numMulWc=nn*(p/nn)
#
# effort=int(numMulWa+numMulWb+numMulWc+p)
# print("Aufwand gelernter Algo: "+str(effort))
# print("n^3:                    "+str(int(n**3)))
