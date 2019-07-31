import murmurhash as mh
import math as mt
import numpy as np


class bloomFilter(object):
    def __init__(self, n, P):
        # self.m = int(-n*mt.log(P, 2)/(mt.log(2, 2)**2))
        # self.K = int(self.m/n*mt.log(2, 2))

        self.m = int(-n*mt.log(P)/(mt.log(2)**2))
        self.K = int(self.m/n*mt.log(2))

        self.mem = np.zeros(self.m, dtype=bool)
        print("Bloom Filter:")
        print("P = ", P)
        print("m = ", self.m)
        print("k = ", self.K)
        return

    def mmhash(self, W, k):
        p = W[0].shape[0]
        hs = np.zeros(p)
        WW = [W[0].copy(), W[1].copy()]
        for i in range(p):
            v = np.concatenate([WW[0][i, :], WW[1][i, :]])
            hs[i] = mh.hash(v.tobytes(), 1) % 1000000
        sorted = np.argsort(hs)
        WW[0] = WW[0][sorted, :]
        WW[1] = WW[1][sorted, :]
        h = mh.hash(np.concatenate([WW[0].flatten(), WW[1].flatten()]).tobytes(), k)
        return h % self.m

    def store(self, W):
        ret = True
        for k in range(self.K):
            h = self.mmhash(W, k)
            if self.mem[h] == False:
                ret = False
            self.mem[h] = True
        return ret


class bloomFilterMatrix(object):
    def __init__(self, n, P):
        self.m = int(-n*mt.log(P)/(mt.log(2)**2))
        self.K = int(self.m/n*mt.log(2))
        self.mem = np.zeros(self.m, dtype=bool)
        return

    def mmhash(self, M, k):
        p = M.shape[0]
        hs = np.zeros(p)
        MM = M.copy()
        for i in range(p):
            hs[i] = mh.hash(MM[i, :].tobytes(), 1) % 1000000
        sorted = np.argsort(hs)
        MM = MM[sorted, :]
        h = mh.hash(MM.flatten().tobytes(), k)
        return h % self.m

    def store(self, M):
        ret = True
        for k in range(self.K):
            h = self.mmhash(M, k)
            if self.mem[h] == False:
                ret = False
            self.mem[h] = True
        return ret


def sortSolution(Wa, Wb, Wc):
    p = Wa.shape[0]
    hs = np.zeros(p)
    for i in range(p):
        v = np.concatenate(Wa[i, :], Wb[i, :])
        hs[i] = mh.hash(v.tobytes(), 1) % 1000000
    sorted = np.argsort(hs)
    WaRet = Wa[sorted, :]
    WbRet = Wb[sorted, :]
    WcRet = Wc[:, sorted]
    return WaRet, WbRet, WcRet


def sortMatrix(M):
    p = M.shape[0]
    hs = np.zeros(p)
    for i in range(p):
        hs[i] = mh.hash(M[i, :].tobytes(), 1) % 1000000
    sorted = np.argsort(hs)
    MM = M[sorted, :]
    return M


if __name__ == '__main__':
    W = np.load("bestSolution_backup_n_3_100Proz_15Mai.npy", allow_pickle=True)
    nn = W[0].shape[1]
    n = int(mt.sqrt(nn))
    p = W[0].shape[0]

    BF = bloomFilter(3*nn*p, 0.0001)

    BF.store(W)

#
