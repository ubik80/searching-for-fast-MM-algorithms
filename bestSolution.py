import numpy as np
import checkSolution as cs
import reduceOps as ro

fc = np.load("solution_3_1117_1562626845.291288_V15.npy", allow_pickle=True)
Wa = fc[0]
Wb = fc[1]
Wc = fc[2]
cs.checkSolutionInt([Wa, Wb, Wc])

A, L = ro.optimizeAddsSubs(Wc, 5000)
R = ro.calcResiduals(A, L, Wc)

len(A)
M = R[2]

latexString = "\\begin{bmatrix}"
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j] == -1:
            latexString += " -1 "
        if M[i, j] == 0:
            latexString += " 0 "
        if M[i, j] == 1:
            latexString += " 1 "
        if j != M.shape[1]-1:
            latexString += "&"
    if i != M.shape[0]-1:
        latexString += "\\\\"
latexString += "\\end{bmatrix}"

print(latexString)
