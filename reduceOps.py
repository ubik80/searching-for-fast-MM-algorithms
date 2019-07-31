import numpy as np


def buildExprList(M, maxLength):
    exprT = [np.array(M[i, :]).flatten() for i in range(M.shape[0])]
    expr = []
    for i in range(len(exprT)):
        if np.sum(np.abs(exprT[i])) > 1:
            expr.append(exprT[i])
    newExprFound = True
    endI = 0
    while newExprFound and len(expr) < maxLength:
        # print(len(expr))
        newExprFound = False
        startI = endI
        endI = len(expr)
        for i in range(startI, endI):
            e = expr[i]
            setPos = np.argwhere(e != 0).flatten()
            if len(setPos) > 2:
                for ii in range(len(setPos)):
                    newExpr = e.copy()
                    newExpr[setPos[ii]] = 0
                    for iii in range(len(expr)):
                        found = True
                        for j in range(len(newExpr)):
                            if newExpr[j] != expr[iii][j]:
                                found = False
                                break
                        if found:
                            break
                    if not found:
                        expr.append(newExpr)
                        newExprFound = True
                        if len(expr) == maxLength:
                            break
            if len(expr) == maxLength:
                break
                # print(newExpr)
    return expr


def rankExpr(L, M):
    profs = np.zeros(len(L))
    for i in range(len(L)):
        reps = 0
        for ii in range(M.shape[0]):
            fits = True
            for j in range(M.shape[1]):
                if L[i][j] != 0 and L[i][j] != M[ii, j]:
                    fits = False
                    break
            if fits:
                reps += 1
        ops = np.max(np.sum(np.abs(L[i]))-1, 0)
        profs[i] = ops*np.max(reps-1, 0)
    return profs


def sortByProfit(p, L):
    sortedIdx = np.argsort(p)
    LL = []
    for i in range(len(sortedIdx)-1, -1, -1):
        if p[sortedIdx[i]] == 0:
            break
        LL.append(L[sortedIdx[i]])
    return LL


def replace(L, M):
    A = np.zeros([M.shape[0], len(L)])
    for i in range(len(L)):
        l = L[i]
        for ii in range(M.shape[0]):
            found = True
            for j in range(M.shape[1]):
                if l[j] != 0 and l[j] != M[ii, j]:
                    found = False
                    break
            if found:
                M[ii, :] -= l
                A[ii, i] = 1
            found = True
            for j in range(M.shape[1]):
                if l[j] != 0 and l[j] != -M[ii, j]:
                    found = False
                    break
            if found:
                M[ii, :] += l
                A[ii, i] = -1
    return list(A)


def shrinkSolution(A, L):
    AMat = np.matrix(A)
    AA = []
    LL = []
    for j in range(AMat.shape[1]):
        if np.sum(np.abs(AMat[:, j])) != 0:
            AA.append(np.array(AMat[:, j].T).flatten())
            LL.append(L[j])
    return np.matrix(AA).T, np.matrix(LL)


def optimizeAddsSubs(M, maxLength):
    assembly = []
    commonExpr = []
    exprList = M
    #recursionLevel = 0
    while True:
        M = np.matrix(exprList)
        exprList = buildExprList(M, maxLength)
        print(len(exprList), end=" ", flush=True)
        profit = rankExpr(exprList, M)
        exprList = sortByProfit(profit, exprList)
        A = replace(exprList, M)
        A, exprList = shrinkSolution(A, exprList)
        if A.shape[0] == 0:
            break
        assembly.append(A)
        commonExpr.append(exprList)
        #recursionLevel += 1
    return assembly, commonExpr


def calcResiduals(A, L, M):
    R = []
    for i in range(len(A)):
        a = np.matrix(A[i])
        l = np.matrix(L[i])
        M = np.matrix(M)
        r = np.matrix(M-a.dot(l))
        R.append(r)
        M = l
    return R


def matVecProd(M, x):
    y = np.zeros(M.shape[0], dtype=int)
    totalOps = 0
    for i in range(M.shape[0]):
        ops = -1
        for j in range(M.shape[1]):
            if M[i, j] != 0 and x[j] != 0:
                ops += 1
                y[i] = 1
        ops = np.maximum(0, ops)
        totalOps += ops
    return y, totalOps


def vecAdd(x, y):
    z = np.zeros(len(x), dtype=int)
    totalOps = 0
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            totalOps += 1
        if x[i] != 0 or y[i] != 0:
            z[i] = 1
    return z, totalOps


def countOptimizedOps(A, L, R):
    if len(A) == 0:
        print("no reduction")
        return -1
    totOps = 0
    x = np.ones(R[0].shape[1])
    Rx, ops = matVecProd(R[len(R)-1], x)
    totOps += ops
    Lx, ops = matVecProd(L[len(L)-1], x)
    totOps += ops
    ALx, ops = matVecProd(A[len(A)-1], Lx)
    totOps += ops
    xx, ops = vecAdd(ALx, Rx)
    totOps += ops
    for i in range(len(A)-2, -1, -1):
        Rx, ops = matVecProd(R[i], x)
        totOps += ops
        Axx, ops = matVecProd(A[i], xx)
        totOps += ops
        xx, ops = vecAdd(Axx, Rx)
        totOps += ops
    return totOps


def countRawOps(M):
    totOps = 0
    for i in range(M.shape[0]):
        totOps += np.maximum(0, np.sum(np.abs(M[i, :]))-1)
    return int(totOps)


#sol = np.load("solution_3_2024_1562198259.6632235_V14.npy", allow_pickle=True)
#Wa, Wb, Wc = sol[0], sol[1], sol[2]
# Wa, Wb, Wc = np.load("manualSolution_n2_p7_2.npy", allow_pickle=True)
# Wa, Wb, Wc = np.round(Wa), np.round(Wb), np.round(Wc)
# M = Wb
#
# A, L = optimizeAddsSubs(M)
# R = calcResiduals(A, L, M)
#
# print(countOptimizedOps(A, L, R))
# print(countRawOps(M))
