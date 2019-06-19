import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

Wa,Wb,Wc,diffs,numOfIters,numOfCycles=np.load("solution_3_1278_1560893173.206902.npy",allow_pickle=True)
Wa,Wb,Wc,diffs,numOfIters,numOfCycles=np.load("solution_3_163_1560840678.386328.npy",allow_pickle=True)

nn=Wc.shape[0]
p=Wc.shape[1]
n=int(np.round(np.sqrt(nn)))

numOfCycles
plt.plot(diffs)
plt.show()

# print("Wa:")
# print(Wa)
# print("Wb:")
# print(Wb)
# print("Wc:")
# print(Wc)
#
# A=np.random.randint(-9,9,[n,n])
# B=np.random.randint(-9,9,[n,n])
# print("Beispiel:")
# print("A:")
# print(A)
# print("B:")
# print(B)
# print("C=A*B exakt:")
# print(A.dot(B))
# a=np.array(A.reshape(n**2),dtype=float)
# b=np.array(B.reshape(n**2),dtype=float)
# c=Wc.dot(Wa.dot(a)*Wb.dot(b))
# C=c.reshape((n,n))
# print("C=Wc*(Wa*b o Wb*b):")
# print(C)
