import numpy as np
import matplotlib.pyplot as plt

fileContent = np.load("solution_3_1366_1565006943.5106518_V19.npy", allow_pickle=True)

diffs = fileContent[4]

maxY = np.max(diffs)-0.2
minY = np.min(diffs)-0.2
plt.rcParams.update({'font.size': 10})
plt.plot(diffs)
plt.xlabel("iteration")
plt.ylabel("| Δ |")
y = [minY, maxY]
ttl = " "
# for j in range(len(jumps)):
#     x = [jumps[j], jumps[j]]
#     plt.plot(x, y, '-r', alpha=0.5)
#     #plt.text(jumps[j]-20, maxY-0.2, heights[j])
#ttl = "jump factor = "+str(jumpFactor)
plt.title(ttl)

picName = 'noJumpsExample1.png'
plt.savefig(picName, dpi=300)
plt.close()
