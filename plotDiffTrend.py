import numpy as np
import matplotlib.pyplot as plt

# fileContent = np.load(
#     "/Users/tillspaeth/Google Drive/V19DiffMap/solution_3_1366_1565006943.5106518_V19.npy", allow_pickle=True)


fileContent = np.load("solution_3_1336_1567508025.3034182_V25.npy", allow_pickle=True)


jumpFactor = fileContent[3]
print("jumpFactor: ", jumpFactor)
diffs = fileContent[4]
jumps = fileContent[5]

maxY = np.max(diffs)-0.2
minY = np.min(diffs)-0.2
plt.rcParams.update({'font.size': 14})
plt.plot(diffs)
plt.xlabel("iteration")
plt.ylabel("| Δ |")
y = [minY, maxY]
ttl = " "
for j in range(len(jumps)):
    x = [jumps[j], jumps[j]]
    plt.plot(x, y, '-r', alpha=0.5)

#plt.savefig("/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung/noJumpsExmpl_2.png", dpi=300)
# plt.close()
