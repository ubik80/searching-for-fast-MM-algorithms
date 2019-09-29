# coding: utf8
import numpy as np

np.set_printoptions(precision=2, suppress=True)


def printM(M, ii=-999, jj=-999, badList=[]):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i == ii and j == jj:
                print("🍨", end='', flush=True)
            elif (i, j) in badList:
                print("💊", end='', flush=True)
            else:
                if M[i, j].astype(int) == 1:
                    print("🍅", end='', flush=True)
                else:
                    print("🍏", end='', flush=True)
        print('')
    return  # printM


def printMM(MA, MB, iA=-999, jA=-999, iB=-999, jB=-999):
    for i in range(MA.shape[0]):
        for j in range(MA.shape[1]):
            if iA == i and jA == j:
                print("🍨", end='', flush=True)
            else:
                if MA[i, j].astype(int) == 1:
                    print("🍅", end='', flush=True)
                else:
                    print("🍏", end='', flush=True)
        print("     ", end='', flush=True)
        for j in range(MB.shape[1]):
            if iB == i and jB == j:
                print("🍨", end='', flush=True)
            else:
                if MB[i, j].astype(int) == 1:
                    print("🍅", end='', flush=True)
                else:
                    print("🍏", end='', flush=True)
        print('')
    return  # printMM
