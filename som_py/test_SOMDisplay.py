import numpy as np
import matplotlib.pyplot as plt
import SOMDisplay

np.random.seed(42)
points = np.random.uniform(size=(2000,2)).astype(np.float32) - 0.5
#scores = np.ones(1000).astype(np.float32)
dists = np.sum(np.abs(points - np.array([0.2, 0.2])), axis=1)

for i in range(5,55,5):
    scores = np.exp(  -dists * i ).astype(np.float32)

    repres = SOMDisplay.create_som_display(points, scores, 3, 3)
    repres = points[repres]
    plt.scatter(points[:,0], points[:,1], marker="o", s=10, c=scores, cmap="Greens")
    plt.scatter(repres[:,0], repres[:,1], marker="x", s=100)
    plt.show()
