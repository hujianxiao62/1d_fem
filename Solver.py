import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh
from Assembler import Assembler
import math


def k(x):
    return 1

def b(x):
    return 0

def c(x):
    return 0

def f(x):
    return math.pi**2*math.cos(math.pi*x/2)/4

def u_exact(x):
    return math.cos(math.pi*x/2)

def u_exact_dif(x):
    return -math.pi/2*math.sin(math.pi*x/2)

def main():

    num_elements = 32 #number of element
    Etype = 2 #order of shape function
    Gpoints = 4 #guassian quadrature points
    x_left = 0 #left boundary of domain
    x_right = 1 #right boundary of domain
    
    alpha0 = 0
    alpha1 = 1
    beta0 = 1
    beta1 = 0
    gamma0 = 1
    gamma1 = -math.pi/2
    
# =============================================================================
#     -d(kdu/dx)/dx +cdu/dx +bu = f
#     alpha0*du(0)/dx + beta0*u(0) = gamma0
#     alpha1*du(1)/dx + beta1*u(1) = gamma1
# =============================================================================

    # Get element list
    mesh = Mesh.uniform(x_left, x_right, num_elements, Etype)

    # Assemble to get K and F    
    model = Assembler(mesh, k, b, c, f, Etype, Gpoints)  
    K = model.K
    F = model.F
    
    #add boundary condition
    F[0] = F[0] - k(x_left)*gamma0/(alpha0+10** -5)
    F[-1] = F[-1] + k(x_right)*gamma1/(alpha1+10** -5)
    K[0, 0] = K[0, 0] - k(x_left)*beta0/(alpha0+10** -5)
    K[-1, -1] = K[-1, -1] + k(x_right)*beta1/(alpha1+10** -5)

    #solve
    u = np.linalg.solve(K, F)
    
    print(F)


    u_e = np.array([u_exact(t) for t in model.mesh.nIndex])   
    # Plot solution
    plt.plot(model.mesh.nIndex, u_e, label= 'u(x)')
    plt.plot(model.mesh.nIndex, u, "k*-", label= 'fem')
    plt.xlabel('1D interval')
    plt.ylabel('displacement')
    plt.legend(loc="best")
    plt.show()
      

    #calculate H1 error and L2 error using 2 point Gaussian Quadrature
    wi = [1.0, 1.0]
    xi = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    xii = [(xi[0]+1)/2, (xi[1]+1)/2] # integral point in interval [0, 1]
    eh_ts = 1/num_elements # element length
    ex = [int((xii[0])/eh_ts), int((xii[1])/eh_ts)] #element index corresponding to integral points
    
    #initialize
    u_fem = [0, 0] #results of u from shape function for two Gaussian point
    u_fem_x = [0 , 0]  #results of u' from shape function for two Gaussian point
    u_ex = [0, 0]  #results of u from real solution for two Gaussian point
    u_ex_x = [0 , 0]  #results of u' from real solution for two Gaussian point
    index = 0

    # calculate Gaussian intergal using the according element's shape function
    for i in ex:

        element = mesh.elements[i]
        xa = element.x_left
        xb = element.x_right
                
        x = xi[index]*(xb-xa)/2 + (xb+xa)/2
        w = wi[index] * (xb-xa)/2
        N1 = element.N(x=x, local_node=1, Etype = 1)
        N1x = element.Nx(x=x, local_node=1, Etype = 1)
        N2 = element.N(x=x, local_node=2, Etype = 1)
        N2x = element.Nx(x=x, local_node=2, Etype = 1)
 
        u_fem[index] = (N1*u[i]+N2*u[i+1])*w
        u_fem_x[index] = (N1x*u[i]+N2x*u[i+1])*w
        u_ex[index] = u_exact(x)*w
        u_ex_x[index] = u_exact_dif(x)*w
        index += index
            
    H1 = ((u_fem_x[0] - u_ex_x[0])**2 + (u_fem_x[1] - u_ex_x[1])**2 +\
         (u_fem[0] - u_ex[0])**2 + (u_fem[1] - u_ex[1])**2)**(1/2)
    
    L2 = ((u_fem[0] - u_ex[0])**2 + (u_fem[1] - u_ex[1])**2)**(1/2)
    
    print ('H1:', H1 , '  ' , 'L2:' , L2)
                    



if __name__ == '__main__':
    main()
