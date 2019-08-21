import numpy as np
from element import Element


class Mesh(object):
    def __init__(self, x, y, elements):
        self.eIndex = x
        self.nIndex = y
        self.elements = elements
        self.num_elements = len(elements)

    @classmethod
    def uniform(cls, x_start, x_end, num_elements, Etype):

        # element index
        eIndex = np.linspace(x_start, x_end, num_elements+1)
        # node index        
        nIndex = np.linspace(x_start, x_end, num_elements*Etype+1)

        # mesh body to arrary of element
        elements = [Element(i, eIndex[i], eIndex[i+1], Etype) for i in range(len(eIndex)-1)]

        return cls(eIndex, nIndex, elements)

