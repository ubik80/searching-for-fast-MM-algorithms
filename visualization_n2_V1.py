import matplotlib.animation as animation
from sklearn.manifold import TSNE
import numpy as np
import checkSolution as cs
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import os
import scipy
import cv2

video = True
DPI = 300

fc = np.load("n2_vis_4.npy", allow_pickle=True)
Wa, Wb, Wc = fc[0], fc[1], fc[2]
diffs = fc[3]
WHist = fc[4]
jmps = WHist

cs.checkSolutionInt([Wa, Wb, Wc])

X = []
for i in range(len(WHist)):
    X.append(np.concatenate((WHist[i][0].flatten(), WHist[i][1].flatten())))
    # X.append(np.array(WHist[i][0].flatten()))
X_emb = TSNE(n_components=2, perplexity=50).fit_transform(X)

plt.figure()
x = []
y = []
z = []
diffsIdx = 0
for i in range(len(X_emb)-1):
    if WHist[i][3] == False:
        x.append(X_emb[i][0])
        y.append(X_emb[i][1])
        z.append(diffs[diffsIdx+1])
        diffsIdx += 1
tcf = plt.tricontourf(x, y, z, 15, cmap='binary', extend='neither', alpha=0.3)
plt.colorbar(tcf)
plt.axis('off')

plt.plot(X_emb[0, 0], X_emb[0, 1], color='green', marker='o', markersize=10)
plt.axis('off')
if video:
    plt.savefig('/Users/tillspaeth/trash/0', dpi=DPI)

oldX = X_emb[0, 0]
oldY = X_emb[0, 1]
for i in range(1, len(X_emb)):
    if WHist[i][3] == False:
        plt.plot((oldX, X_emb[i, 0]), (oldY, X_emb[i, 1]), color='blue',
                 marker='.', markersize=2, linestyle='-', linewidth=0.5)
        if WHist[i-1][3] == True:
            plt.plot(oldX, oldY, markersize=4, color='magenta', marker='.')
    else:
        plt.plot(oldX, oldY, markersize=4, color='magenta', marker='.')
        plt.plot((oldX, X_emb[i, 0]), (oldY, X_emb[i, 1]), color='magenta',
                 marker='.', markersize=4, linestyle='-', linewidth=1.0)
    # plt.plot(X_emb[i, 0], X_emb[i, 1], color=cmap(diffs[i]),
    #          marker='.', markersize=50, linestyle='-', linewidth=1.0, zorder=-100, alpha=1.0)
    oldX = X_emb[i, 0]
    oldY = X_emb[i, 1]
    plt.title(str(i))
    plt.axis('off')
    if video:
        plt.savefig('/Users/tillspaeth/trash/'+str(i), dpi=DPI)

plt.plot(X_emb[len(X_emb)-1, 0], X_emb[len(X_emb)-1, 1], color='red', marker='o', markersize=10)
if video:
    plt.savefig('/Users/tillspaeth/trash/'+str(i+1), dpi=DPI)

plt.title("")
if video:
    plt.savefig('/Users/tillspaeth/trash/forDoc', dpi=DPI)

if video:
    img = cv2.imread('/Users/tillspaeth/trash/0.png')
    frame_size = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter('output.avi', fourcc, 3.0, (frame_size[1], frame_size[0]))
    for i in range(len(X_emb)+1):
        img = cv2.imread('/Users/tillspaeth/trash/'+str(i)+'.png')
        frame_size = img.shape
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
