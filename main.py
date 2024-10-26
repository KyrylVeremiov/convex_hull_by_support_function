# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
# https://docs.scipy.org/doc/scipy/reference/spatial.html
import scipy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def c(X,p):
    f=max(X,key=lambda x: np.dot(x,p))
    return {"val":np.dot(f,p),"f":f}

def convex_hall(X,n):
    psi_s=[]
    for a in np.arange(0,n):
        psi_s.append([np.cos(a*2*np.pi/n),np.sin(a*2*np.pi/n)])
    psi_s.append(psi_s[0])
    psi_s=np.array(psi_s)

    hull_approx=[]

    c_prev=0
    for i in range(psi_s.shape[0]):
        c_curr=c(X,psi_s[i])
        if i!=0:
            hull_approx.append(
                scipy.linalg.solve(
                    [psi_s[i-1],psi_s[i]],[c_prev["val"],c_curr["val"]]))
        c_prev=c_curr

    return np.array(hull_approx)


N_psi=10
# N=30
N=70
SEED=10


np.random.seed(SEED)
points = np.random.rand(N, 2)   # 30 random points in 2-D
# hull = ConvexHull(points)

plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#
# plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
# plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
# plt.show()


hull_approx=convex_hall(points, N_psi)
plt.plot(hull_approx[:,0], hull_approx[:,1], 'r--', lw=2)
plt.plot([hull_approx[:,0][0],hull_approx[:,0][-1]],[hull_approx[:,1][0],hull_approx[:,1][-1]], 'r--', lw=2)
plt.show()

#############################################################################
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

N_psi_s=[3,4,5,10,30,50,100]

def init():
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.plot(points[:, 0], points[:, 1], 'bo')
    return ln,

def update(frame):
    ax.clear()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.plot(points[:, 0], points[:, 1], 'bo')
    hull_approx = convex_hall(points, frame)
    ax.plot(hull_approx[:, 0], hull_approx[:, 1], 'r--', lw=2)
    ax.plot([hull_approx[:, 0][0], hull_approx[:, 0][-1]], [hull_approx[:, 1][0], hull_approx[:, 1][-1]], 'r--', lw=2)
    return ln,

ani = FuncAnimation(fig, update, frames=N_psi_s,
                    init_func=init, blit=True,interval=1000)
# plt.show()
# plt.savefig("a.gif")
ani.save("res.gif")