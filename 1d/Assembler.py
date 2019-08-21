import numpy as np
import math


class Assembler(object):    

    def __init__(self, mesh, kk, b, c, f, Etype, Gpoints):
        self.mesh = mesh        
        self.K = np.zeros((self.mesh.num_elements*Etype+1, self.mesh.num_elements*Etype+1))
        self.F = np.zeros_like(np.zeros(self.mesh.num_elements*Etype+1))
        
        # set Gaussian Quadrature points
        if Gpoints == 2:
            wi = [1.0, 1.0]
            xi = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        elif Gpoints == 3:
            wi = [5/9, 8/9, 5/9]
            xi = [-math.sqrt(3/5), 0, math.sqrt(3/5)]
        elif Gpoints == 4:
            wi = [0.347855, 0.652145, 0.652145, 0.347855]
            xi = [-0.861136, -0.339981, 0.339981, 0.861136]
        else:
            raise ValueError('choose Gaussian quadrature points: 2-4')
        
        # Loop over elements.

        for element in self.mesh.elements:

            #index of the first node of the element in Gobal Stiffness Metrix
            eStart = element.index*Etype
            xa = element.x_left
            xb = element.x_right

            # Loop over shape functions Ni.
            for i in range(Etype+1):

                # Loop over shape functions Nj.
                for j in range(Etype+1):

                    # Loop over GQ points.
                    for k in range(Gpoints):
                        #transfer interval
                        x = xi[k]*(xb-xa)/2 + (xb+xa)/2
                        w = wi[k] * (xb-xa)/2

                        # get shape function 
                        Ni = element.N(x=x, local_node=i, Etype=Etype)
                        Nj = element.N(x=x, local_node=j, Etype=Etype)
                        Ni_x = element.Nx(x=x, local_node=i, Etype=Etype)
                        Nj_x = element.Nx(x=x, local_node=j, Etype=Etype)

                        # get local stiffness and assamble it to gobal 
                        if Etype <= 2:
                            k_e_ij = Ni_x * kk(x) * Nj_x + Ni * b(x) * Nj + Ni * c(x) * Nj_x  
                            self.K[eStart+i, eStart+j] += w * k_e_ij
                        else:
                            raise ValueError('Choose Element Type 1 or 2')

            for i in range(Etype+1):

                for k in range(Gpoints):
                    
                    x = xi[k]*(xb-xa)/2 + (xb+xa)/2
                    w = wi[k] * (xb-xa)/2
                    Ni = element.N(x=x, local_node=i, Etype = Etype)

                    # get force on local node and assamble it to gobal
                    f_e_j = Ni * f(x)
                    self.F[eStart+i] += w * f_e_j
                    
