import numpy as np
import matplotlib.pyplot as plt
import SOMDisplay

points = np.random.uniform(size=(20000,2)).astype(np.float32)
#scores = np.ones(1000).astype(np.float32)
dists = np.sum(np.abs(points - np.array([0.5, 0.5])), axis=1)
scores = np.exp(  -dists * 10 ).astype(np.float32)

repres = SOMDisplay.create_som_display(points, scores, 4, 4)
repres = points[repres]
plt.scatter(points[:,0], points[:,1], marker="o", s=10, c=scores, cmap="Greens")
plt.scatter(repres[:,0], repres[:,1], marker="x", s=100)
plt.show()
