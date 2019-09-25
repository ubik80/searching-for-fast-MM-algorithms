import matplotlib.animation as animation
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import checkSolution as cs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import os
import scipy
import cv2

DPI = 300

fc1 = np.load("n2_vis_V2_1.npy", allow_pickle=True)
fc2 = np.load("n2_vis_V2_2.npy", allow_pickle=True)
fc3 = np.load("n2_vis_V2_3.npy", allow_pickle=True)
fc4 = np.load("n2_vis_V2_4.npy", allow_pickle=True)
WHist1 = fc1[4]
WHist2 = fc2[4]
WHist3 = fc3[4]
WHist4 = fc4[4]

WHistComb = WHist1+WHist2+WHist3+WHist4
len(WHistComb)

X = []
for i in range(len(WHistComb)):
    X.append(np.concatenate((WHistComb[i][0].flatten(), WHistComb[i][1].flatten())))

pca = PCA(n_components=32)
pca.fit(X)
np.sum(pca.explained_variance_ratio_)
X = pca.fit_transform(X)

X = TSNE(n_components=2).fit_transform(X)  # , perplexity=50

plt.figure()
# x = []
# y = []
# z = []
# for i in range(len(X)-1):
#     if WHistComb[i+1][4] > 0.0:
#         x.append(X[i][0])
#         y.append(X[i][1])
#         z.append(WHistComb[i+1][4])
# tcf = plt.tricontourf(x, y, z, 15, cmap='binary', extend='neither', alpha=0.3)
# plt.colorbar(tcf)
# plt.axis('off')
plt.plot(X[0, 0], X[0, 1], color='green', marker='o', markersize=10, zorder=100)
plt.axis('off')

colors = ["blue", "green", "black", "cyan"]
clrIdx = 0

oldX = X[0, 0]
oldY = X[0, 1]
for i in range(1, len(X)):
    color = colors[clrIdx]
    if WHistComb[i][4] > 0.0:
        plt.plot((oldX, X[i, 0]), (oldY, X[i, 1]), color=color,
                 marker='.', markersize=2, linestyle='-', linewidth=0.5)
        if WHistComb[i-1][4] == 0:
            plt.plot(oldX, oldY, markersize=4, color='magenta', marker='.')
    elif WHistComb[i][4] == 0:
        plt.plot(oldX, oldY, markersize=4, color='magenta', marker='.')
        plt.plot((oldX, X[i, 0]), (oldY, X[i, 1]), color='magenta',
                 marker='.', markersize=4, linestyle='-', linewidth=1.0)
    elif WHistComb[i][4] == -2:
        plt.plot(X[i, 0], X[i, 1],
                 color='red', marker='o', markersize=10, zorder=-100)
        clrIdx += 1
    oldX = X[i, 0]
    oldY = X[i, 1]
    plt.title(str(i))
    plt.axis('off')


plt.title("")
plt.savefig('/Users/tillspaeth/trash/forDoc', dpi=DPI,  bbox_inches='tight')
