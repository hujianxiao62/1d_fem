# =============================================================================
# code for MEE615 PROJ3
# Diffusion coef #util 60
# Forcing #util 231
# essential bc #util 33
# neumann bc #util 16
# reference finite element #util 354
# Quadrature rule #util 91 158
# initial condition #util 6
# initial data #util 514
# Point source #util 220
# =============================================================================


from util import *

meshNum=64

domainSize=[0,1]

shapeOrder=1  #shape functiion order = 1, higher is not supported

gaussianPoints=3 # Gaussian Quadrature points < 4

caseID=2  # benchmark:1, application 2

steps=10

time_start=0

time_end=1

alpha=1

T, PM,LLM,t ,L2 = FEM2dHeat (meshNum,domainSize, shapeOrder, gaussianPoints, caseID, steps, time_end, time_start, alpha)

print('---L2 error---')
print(L2)

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

ta ,te = getSolution(T,PM,LLM,shapeOrder,meshNum,t)

fig = plt.figure()
ax0 = plt.subplot(1,2,1)
ax1 = plt.subplot(122)
taa = ta[:,:,0]
tee = te[:,:,0]
field1 = ax0.pcolor(taa)
ax0.title.set_text('FEM')
field2 = ax1.pcolor(tee)
ax1.title.set_text('EXACT')

def make_frame(t):
    taa = ta[:,:,t]
    tee = te[:,:,t]
    field1 = ax0.pcolor(taa)
    field2 = ax1.pcolor(tee)
    return field1, field2

ani=animation.FuncAnimation(fig,func=make_frame,frames=range(11),blit=True)
plt.show()

