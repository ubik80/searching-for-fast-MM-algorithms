import numpy as np
import backprop as biM
import multiprocessing as mp
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
            nue = np.random.rand()*0.09+0.01
            print(".", end='', flush=True)
            iters = biM.backprop(Wa, Wb, Wc, 50000000, nue, 0.000001)
            if iters > 0 and np.sum(np.isnan(Wa)) == 0:
                np.save("findStartValues_5_"+str(p)+"_id_"+str(id)+"_V1", [Wa, Wb, Wc, nue])
                success = True
                print("p = ", p, " found.")
    return  # goSearch


if __name__ == '__main__':
    numOfProc = int(mp.cpu_count())*0+4
    print("Anzahl Prozessoren: ", numOfProc)

    procs = [mp.Process(target=goSearch, args=[0]) for i in range(numOfProc)]

    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()
