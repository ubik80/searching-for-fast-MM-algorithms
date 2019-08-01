import numpy as np
import backprop as biM
import multiprocessing as mp
import uuid
np.set_printoptions(precision=2, suppress=True)

n = 5
nn = 25


def goSearch(id):
    print("process ", str(id), " started.")
    for p in range(125, 0, -1):
        success = False
        while not success:
            Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
            Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
            Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
            nueAB = 0.05
            nueC = 0.15
            tol = 0.1
            iters = 999
            while tol > 0.00001 and iters > 0:
                iters = biM.backpropNue(Wa, Wb, Wc, 50000000, tol, nueAB, nueC)
                print("p:"+str(p)+" tol:"+str(tol)+" iters:"+str(iters))
                tol *= 0.99
            if iters > 0 and np.sum(np.isnan(Wa)) == 0:
                np.save("findStartValues_5_"+str(p)+"_id_"+str(id)+"_" +
                        str(int(uuid.uuid4()))+"_V1", [Wa, Wb, Wc, nueAB, nueC, tol])
                success = True
                print("solution for p = ", p, " found.")
    return  # goSearch


if __name__ == '__main__':
    numOfProc = int(mp.cpu_count())*0+4
    print("Anzahl Prozessoren: ", numOfProc)

    procs = [mp.Process(target=goSearch, args=[i]) for i in range(numOfProc)]

    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()
