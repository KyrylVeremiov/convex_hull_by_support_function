# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
# https://docs.scipy.org/doc/scipy/reference/spatial.html
import scipy
import sympy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def c(cF,psi):
    return cF.subs(psi1,psi[0]).subs(psi2,psi[1])

def convex_hall(F, n):
    psi_s=[]
    for a in np.arange(0,n):
        psi_s.append([np.cos(a*2*np.pi/n),np.sin(a*2*np.pi/n)])
    psi_s.append(psi_s[0])
    psi_s=np.array(psi_s)

    hull_approx=[]

    c_prev=0
    for i in range(psi_s.shape[0]):
        c_curr=c(F, psi_s[i])
        if i!=0:
            hull_approx.append(
                scipy.linalg.solve(
                    [psi_s[i-1],psi_s[i]],[float(c_prev),float(c_curr)]))
        c_prev=c_curr

    return np.array(hull_approx)

###########################################################
N_psi=200
# N=30
N=70
SEED=10


psi1,psi2=sympy.symbols('psi1 psi2')

# sphere set
x1_sphere_center=4
x2_sphere_center=5
r=10
cF_sphere= psi1 * x1_sphere_center + psi2 * x2_sphere_center + r * sympy.sqrt(psi1 * psi1 + psi2 * psi2)

# rectangle set
x1_rectangle_center=4
x2_rectangle_center=6
r_1_rectangle=3
r_2_rectangle=5
cF_square= (psi1 * x1_rectangle_center + psi2 * x2_rectangle_center
            + abs(psi1)*r_1_rectangle+abs(psi2)*r_2_rectangle)

cF=cF_square+cF_sphere
hull_approx=convex_hall(cF, N_psi)
plt.plot(hull_approx[:,0], hull_approx[:,1], 'r--', lw=2)
plt.plot([hull_approx[:,0][0],hull_approx[:,0][-1]],[hull_approx[:,1][0],hull_approx[:,1][-1]], 'r--', lw=2)
plt.show()

#############################################################################
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = ax.plot([], [], 'ro')

N_psi_s=[3,4,5,10,30,50,100,200]

def init():
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 1.5)
    return ln,

def update(frame):
    ax.clear()
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 1.5)

    ax.set_aspect('equal')
    hull_approx = convex_hall(cF, frame)

    ax.plot(hull_approx[:, 0], hull_approx[:, 1], 'r--', lw=2)
    ax.plot([hull_approx[:, 0][0], hull_approx[:, 0][-1]], [hull_approx[:, 1][0], hull_approx[:, 1][-1]], 'r--', lw=2)
    return ln,

ani = FuncAnimation(fig, update, frames=N_psi_s,
                    init_func=init, blit=True,interval=1000)
# plt.show()
# plt.savefig("a.gif")
ani.save("res.gif")