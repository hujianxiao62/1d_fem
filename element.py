import numpy as np


class Element(object):

    def __init__(self, index, x_left, x_right, Etype):
        self.Etype = Etype
        self.num_nodes = Etype+1
        self.index = index
        self.x_left = x_left
        self.x_right = x_right
        self.h = x_right - x_left

    #shape function
    def N(self, x, local_node, Etype):

        x = np.asarray(x)
        phi = np.zeros_like(x)
        
        if Etype == 1:

            r = self.x_right
            l = self.x_left
            if local_node == 0:
                phi += (x - r) / (l - r)
            elif local_node == 1:
                phi += (x - l) / (r - l)
            elif local_node == 2:
                phi =0
        
        elif Etype == 2:
            
            r = self.x_right
            l = self.x_left
            m = (r+l)/2
            if local_node == 0:
                phi += (x-m)*(x-r) / ((l-m)*(l-r))
            elif local_node == 1:
                phi += (x-l)*(x-r) / ((m-l)*(m-r))
            elif local_node == 2:
                phi += (x-l)*(x-m) / ((r-l)*(r-m))                   
        else:
            raise ValueError("shape function order: 1 or 2")

        return phi

    #derivitive of shape function
    def Nx(self, x, local_node, Etype):

        x = np.asarray(x)
        phi_x = np.zeros_like(x)
        
        if Etype == 1:

            r = self.x_right
            l = self.x_left
            if local_node == 0:
                phi_x += 1 / (l - r)
            elif local_node == 1:
                phi_x += 1 / (r - l)
            elif local_node == 2:
                phi_x += 0
                
        elif Etype == 2:
            r = self.x_right
            l = self.x_left
            m = (r+l)/2
            if local_node == 0:
                phi_x += (2*x-m-r) / ((l-m)*(l-r))
            elif local_node == 1:
                phi_x += (2*x-l-r) / ((m-l)*(m-r))
            elif local_node == 2:
                phi_x += (2*x-m-l) / ((r-l)*(r-m))     
        else:
            raise ValueError("shape function order: 1 or 2")

        return phi_x
