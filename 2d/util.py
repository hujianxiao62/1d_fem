import numpy
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time as tm
def initialT(caseID,T,PM):

    if 1 == caseID:
        T[:,0]=numpy.cos(PM[:,0] - PM[:,1])

    elif 2 == caseID:
        T[:,0]=0
    
    return T

def naturalBC(cx=None,cy=None,x=None,y=None,caseID=None,h=None,time=None):

    if 1 == caseID:
        if abs(cy - h[1]) < 1e-05:
            nv=-x*numpy.sin(x - 1 + time)

        elif abs(cx - h[0]) < 1e-05:
            nv=-y*numpy.sin(-y + time)

        else:
            nv=0

    elif 2 == caseID:
            nv=0

    return nv

def essentialBC(cx=None,cy=None,xe=None,ye=None,K=None,F=None,caseID=None,h=None,node=None,time=None):

    if 1 == caseID:
        if abs(cy - h[0]) < 1e-04:
            BC=numpy.cos(xe + [time,time])
            isBC = True

        elif abs(cx - h[1]) < 1e-04:
            BC=numpy.cos([1,1] - ye + [time,time])
            isBC = True
        else:
            BC=[0,0]
            isBC = False
    elif 2 == caseID:
            BC=[0,0]
            isBC = True
            
    if isBC:
        for i in range(2):
            K[node[i]-1,:]=0

            K[node[i]-1,node[i]-1]=1

            F[node[i]-1]=BC[i]
    
    return K,F

def Kappa(x,y,caseID):
    
    kappa = numpy.zeros((2,2))
    if 1 == caseID:
        kappa=[[y,0],[0,x]]

    elif 2 == caseID:
        if x >= 0.5:
            kappa=[[1,0],[0,1]]

        else:
            kappa=[[10,0],[0,10]]
    
    return kappa


def Jacobian(x,y,qp,GN):
    GNtmp = numpy.array(GN)
    GNeta=GNtmp[:,qp-1,0]

    GNxi=GNtmp[:,qp-1,1]

    GN=numpy.array([GNeta.T,GNxi.T])
    xbar=numpy.array([x,y]).T

    J=numpy.matmul(GN, xbar)

    return numpy.linalg.det(J)

    

def Gaussian2D(n):

    gp=numpy.zeros(n)  

    gWeight=numpy.zeros(n)  

    if (n == 2):
        gp[0]=- 1 / math.sqrt(3)

        gp[1]=1 / math.sqrt(3)

        gWeight[0]=1

        gWeight[1]=1

    elif (n == 3):
        gp[0]=- math.sqrt(3 / 5)

        gp[1]=0

        gp[2]=math.sqrt(3 / 5)

        gWeight[0]=5 / 9

        gWeight[1]=8 / 9

        gWeight[2]=5 / 9

    elif (n == 4):
        tmp2=math.sqrt(3 / 7 - (2 / 7*math.sqrt(6 / 5)))

        tmp1=math.sqrt(3 / 7 + (2 / 7*math.sqrt(6 / 5)))

        gp[0]=- tmp1

        gp[1]=- tmp2

        gp[2]=tmp2

        gp[3]=tmp1

        tmp3=(18 + math.sqrt(30)) / 36

        tmp4=(18 - math.sqrt(30)) / 36

        gWeight[0]=tmp4

        gWeight[1]=tmp3

        gWeight[2]=tmp3

        gWeight[3]=tmp4
    
    gp2=numpy.zeros((n**2,2))  
    gw2=numpy.zeros((n**2,1))  

    for i in range(n):
        tmp = gp[i]
        gp2[n*i:(i+1)*n,0]=tmp

        gp2[n*i:(i+1)*n,1]=gp[0:n+1]

        gw2[n*i:(i+1)*n,0]=gWeight[i]*gWeight[0:n+1]

    
    return gp2,gw2

def Gaussian1D(n):

    gp=numpy.zeros(n)  

    gWeight=numpy.zeros(n)  

    if (n == 2):
        gp[0]=- 1 / math.sqrt(3)

        gp[1]=1 / math.sqrt(3)

        gWeight[0]=1

        gWeight[1]=1

    elif (n == 3):
        gp[0]=- math.sqrt(3 / 5)

        gp[1]=0

        gp[2]=math.sqrt(3 / 5)

        gWeight[0]=5 / 9

        gWeight[1]=8 / 9

        gWeight[2]=5 / 9

    elif (n == 4):
        tmp2=math.sqrt(3 / 7 - (2 / 7*math.sqrt(6 / 5)))

        tmp1=math.sqrt(3 / 7 + (2 / 7*math.sqrt(6 / 5)))

        gp[0]=- tmp1

        gp[1]=- tmp2

        gp[2]=tmp2

        gp[3]=tmp1

        tmp3=(18 + math.sqrt(30)) / 36

        tmp4=(18 - math.sqrt(30)) / 36

        gWeight[0]=tmp4

        gWeight[1]=tmp3

        gWeight[2]=tmp3

        gWeight[3]=tmp4

    return gp,gWeight

def rightHandSide(x,y,caseID,tt,elementSize):

    t=tt*numpy.ones(len(x))
    center = numpy.zeros(len(x))
    rhs = numpy.zeros(len(x))
    
    for i in range(len(x)):
        if (abs(x[i] - 0.5) < elementSize and abs(y[i] - 0.5) < elementSize):
            center[i] = 1
            
    if caseID == 1:
        rhs=- numpy.sin(x - y + t) + numpy.multiply((x + y),numpy.cos(x - y + t))

    elif caseID == 2:
        if tt <= 0.01:
            rhs = (1+100*t)*center
        else:
            rhs = 2*t*center
    return rhs


def getShape2D(G):
        
    qpoints=len(G)

    N=numpy.zeros((4,qpoints))    
    dN=numpy.zeros((4,qpoints,2)) 
    
    GG=numpy.asarray(G)
    
    eta=GG[:,0]

    xi=GG[:,1]

    for i in range(qpoints):
        N[0][i]=(eta[i] - 1) *(xi[i] - 1) / 4

        N[1][i]=-(eta[i] + 1) *(xi[i] - 1) / 4

        N[2][i]=(eta[i] + 1) *(xi[i] + 1) / 4

        N[3][i]=-(eta[i] - 1) *(xi[i] + 1) / 4

        dN[0][i][0]=(xi[i] - 1) / 4

        dN[1][i][0]=-(xi[i] - 1) / 4
        
        dN[2][i][0]=(xi[i] + 1) / 4

        dN[3][i][0]=-(xi[i] + 1) / 4

        dN[0][i][1]=(eta[i] - 1) / 4

        dN[1][i][1]=-(eta[i] + 1) / 4
        
        dN[2][i][1]=(eta[i] + 1) / 4

        dN[3][i][1]=-(eta[i] -1) / 4

    return N,dN

def getShape1D(qpoints):

    deg = 1
    
    N=[[0 for x in range(len(qpoints))] for y in range(deg+1)] 
    dN=[[0 for x in range(len(qpoints))] for y in range(deg+1)] 

    for i in range(len(qpoints)):
        x=qpoints[i]
        N[0][i]= (1-x)/2
        N[1][i]= (1+x)/2
        dN[0][i]= -0.5
        dN[1][i]= 0.5
    return N,dN

def getBC(nelem,deg,PM,BM,caseID,h,gw1,gp1,J1,N1,K,F,time):
    
    xe = numpy.zeros((deg*nelem,2,4))
    ye = numpy.zeros((deg*nelem,2,4))
    cx = numpy.zeros((deg*nelem,4))
    cy = numpy.zeros((deg*nelem,4))
    node = numpy.zeros((2,deg*nelem,4)).astype(int)
    for face in range(4):
        for e in range(deg*nelem):
            node[:,e,face]=BM[e:e + deg+1,face].astype(int)
            
            xs=PM[node[0,e,face]-1,0]

            ys=PM[node[0,e,face]-1,1]

            xf=PM[node[deg,e,face]-1,0]

            yf=PM[node[deg,e,face]-1,1]

            xe[e,:,face]=[xs,xf]

            ye[e,:,face]=[ys,yf]

            cy[e,face]=(yf + ys) / 2

            cx[e,face]=(xf + xs) / 2

            for j in range(deg + 1):
                NBClocal=0

                for g in range(len(gp1)):
                    x=cx[e,face] + gp1[g]*J1

                    y=cy[e,face] + gp1[g]*J1

                    NBClocal=NBClocal - naturalBC(cx[e,face],cy[e,face],x,y,caseID,h,time) * gw1[g] * N1[j][g] * J1         

                F[node[j,e,face]-1]=F[node[j,e,face]-1] + NBClocal

    for face in range(4):
        for e in range(deg*nelem):
            K,F=essentialBC(cx[e,face],cy[e,face],xe[e,:,face],ye[e,:,face],K,F,caseID,h,node[:,e,face],time)

    return K,F

def localM(N,E,h,N2,gdeg):

    nelem=[i for i in range(E**2)]
    
    LM=numpy.zeros((N*4,E**2))

    node=E*N+1

    step=(h[1]-h[0])/E/N

    space=[i*step for i in range(node)]

    PM=numpy.zeros(((E*N+1)**2,2))

    for i in range(E+1):
        PM[node*i:node*(i+1),0]=space
        tmp = space[i];
        PM[node*i:node*(i+1),1]=tmp


    for i in nelem:
        tmp = int(i/E)
        LM[0][i]= i +1 +tmp
        LM[1][i]= i+2+tmp
        LM[2][i]= E+i+3+tmp
        LM[3][i]= E+i+2+tmp


    BM=numpy.zeros((N*E+1,4))

    for i in range(N*E+1):
        tmp = int(i/E)
        BM[i][0]= i*E+i+1
        BM[i][1]= (i+1)*E+i+1
        BM[i][2]= i+1
        BM[i][3]= (E*N+1)**2-E+i

    X=numpy.zeros((E**2,gdeg ** 2))

    Y=numpy.zeros((E**2,gdeg ** 2))

    for elem in nelem:
        for q in range(gdeg ** 2):
            tmp1 = 0
            tmp2 = 0
            for i in range(4):
                tmp = int(LM[i,elem])-1
                tmp1=tmp1 + N2[i,q]*PM[tmp,0]
                tmp2=tmp2 + N2[i,q]*PM[tmp,1]
            
            X[elem,q]=tmp1
            Y[elem,q]=tmp2
    
    return LM,PM,BM,X,Y

def assemble(nelem,gdeg,caseID,xfun,yfun,LM,gw2,J2,DN2,N2,time,elementsize):

    numNode=(nelem + 1) ** 2

    klocal=numpy.zeros((4,4)).astype(int)

    flocal=numpy.zeros((4,4))

    mlocal=numpy.zeros((4,4)).astype(int)

    K=numpy.zeros((numNode,numNode))

    F=numpy.zeros(numNode)

    M=numpy.zeros((numNode,numNode))
    
    LM = LM.astype(int)
    
    for k in range(nelem ** 2):
        if time == 0:
            for qu in range(gdeg ** 2):

                kappa=Kappa(xfun[k,qu],yfun[k,qu],caseID)

                klocal=klocal + gw2[qu] * numpy.matmul( numpy.matmul( DN2[:,qu,:],  kappa ) , DN2[:,qu,:].T)

            for i in range(gdeg ** 2):
                mlocal= mlocal + gw2[i] * numpy.outer( N2[:,i] , N2[:,i].T ) * J2
        
        flocal = J2* numpy.matmul( numpy.matmul(N2 , numpy.diag(gw2.flatten()) ) , rightHandSide(xfun[k,:].T,yfun[k,:].T,caseID,time,elementsize) );
        
        
        for i in range(4):
            if time == 0:
                for j in range(4):
                    K[LM[i,k]-1,LM[j,k]-1]=K[LM[i,k]-1,LM[j,k]-1] + klocal[i,j]

                    M[LM[i,k]-1,LM[j,k]-1]=M[LM[i,k]-1,LM[j,k]-1] + mlocal[i,j]

            F[LM[i,k]-1]=F[LM[i,k]-1] + flocal[i]

        klocal=numpy.zeros((4,4)).astype(int)

        mlocal=numpy.zeros((4,4)).astype(int)

    
    return K,F,M

def errorL(nelem=None,caseID=None,gw2=None,x=None,y=None,J=None,N=None,T=None,LM=None,time=None,*args,**kwargs):
    L=0
    iter=len(LM)
    Telem=numpy.zeros(iter)

    for e in range(nelem ** 2):
        for gp in range(len(gw2)):
            for j in range(iter):
                Telem[j]=T[LM[j,e].astype(int)-1]

            p=numpy.cos(x[e,gp]-y[e,gp]+time)

            L=L + gw2[gp]*J * (p-numpy.matmul(N[:,gp].T,Telem))**2

    L=math.sqrt(L)

    return L

def getSolution(T,PM,LM,deg,nelem, t):
    
    x= numpy.zeros((4*deg, nelem *nelem))
    y= numpy.zeros((4*deg, nelem ** 2))
    analT= numpy.zeros((4*deg, nelem ** 2))
    exacT= numpy.zeros((4*deg, nelem ** 2))
    ta= numpy.zeros((nelem, nelem, len(T[0])))
    te= numpy.zeros((nelem, nelem, len(T[0])))
    
    for j in range (len(t)):
        for e in range(nelem ** 2):
            for i in range(4*deg):
                tmp=LM[i,e].astype(int)

                x[i,e]=PM[tmp-1,0]

                y[i,e]=PM[tmp-1,1]

                analT[i,e]=T[tmp-1, j]

                exacT[i,e]=numpy.cos(x[i,e]-y[i,e]+t[j])
                
        ta[:,:,j] = analT[0,:].reshape(nelem,nelem)
        te[:,:,j] = exacT[0,:].reshape(nelem,nelem)
    

    return ta, te

def FEM2dHeat (elements,h, shapeD, gdeg, caseID, numtimesteps, tfinal, pstart, alpha):
    start = tm.time()
    
    t=[ i*tfinal/numtimesteps for i in range(numtimesteps+1)]

    delt=tfinal / numtimesteps
    
    numNode=(elements*shapeD+1) ** 2
    
    gp1,gw1=Gaussian1D(gdeg)
    
    N1,DN1=getShape1D(gp1)
    
    gp2,gw2=Gaussian2D(gdeg)
    
    N2,DN2=getShape2D(gp2)
    
    LLM,PM,BM,X,Y=localM(shapeD,elements,h,N2,gdeg)
    
    J1=((h[1] - h[0]) / elements) / 2
    
    J2=Jacobian(PM[LLM[:,0].astype(int)-1,0],PM[LLM[:,0].astype(int)-1,1],1,DN2)
    
    K=numpy.zeros((numNode,numNode))
    
    F=numpy.zeros((numNode,numtimesteps+1))
    
    T=numpy.zeros((numNode,numtimesteps+1))
    
    M=numpy.zeros((numNode,numNode))
    
    T=initialT(caseID,T,PM)
    
    elementsize=(h[1] - h[0]) / elements
    
    
    K,F[:,0],M=assemble(elements,gdeg,caseID,X,Y,LLM,gw2,J2,DN2,N2,0,elementsize)
    
    K,F[:,0]=getBC(elements,shapeD,PM,BM,caseID,h,gw1,gp1,J1,N1,K,F[:,0],0)
    
    for timestep in range(1,numtimesteps + 1):
        time=t[timestep]
    
        __,F[:,timestep],__=assemble(elements,gdeg,caseID,X,Y,LLM,gw2,J2,DN2,N2,time,elementsize)
    
        __,F[:,timestep]=getBC(elements,shapeD,PM,BM,caseID,h,gw1,gp1,J1,N1,K,F[:,timestep],time)
    
        T[:,timestep]=numpy.linalg.solve(M+alpha*delt*K , alpha*delt*F[:,timestep]+numpy.matmul(M,T[:,timestep-1]))
    
    print("--- %s seconds ---" % (tm.time() - start))
    
    L=errorL(elements,caseID,gw2,X,Y,J2,N2,T[:,timestep],LLM,time)
    
    return T, PM,LLM,t ,L
