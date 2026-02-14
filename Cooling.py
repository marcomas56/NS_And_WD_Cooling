import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import animation
import numpy.polynomial.polynomial as poly
import scipy.sparse
import scipy.sparse.linalg
from Emisividad import QTOTAL
from CV import cvtot
import time

'''
Cooling of NS
Yeray Antón y Marco Mas
'''

'''
First we request the mass of the star we want to simulate,
and if we want to save the animation of the cooling process.
'''

print('Select the mass of the star (Select 1, 2 or 3) it may takes 2 minutes:')
print('1: 1.1 M')
print('2: 1.4 M')
print('3: 1.7 M')
aux1 = input()

if aux1 == '1':
    archivo = 'PL200-1.10.DAT'

elif aux1 == '2':
    archivo = 'PL200-1.40.DAT'

elif aux1 == '3':
    archivo = 'PL200-1.70.DAT'

else:
    print('Wrong value, write only 1, 2 or 3')


print('You want to save the animation (It may takes 5 minute):')
print('1: Yes')
print('2: No')
aux2 = input()

if aux2 == '1':
    g = True
elif aux2 == '2':
    g = False
else:
    print('Wrong value, write only 1 or 2')



'''
First we define some normalization constants, since we are going to work with very large
magnitudes, it is convenient to reduce them to make them more manageable numerically.
'''

norm    = 10**(30)
normr   = 10**(6)
normcv  = 10**(27)
normK   = 10**(33)
normt   = 10**(6)
normeps = 10**(30)
normT   = 10**(9)



'''
Here we define several physical constants that we will need.
'''


mn = 1.675*10**(-27)   #SI
mp = 1.673*10**(-27)   #SI
me = 9.11*10**(-31)    #SI
n0 = 0.16*10**(39)     
hb = 1.054*10**(-34)   #SI
c = 3*10**8            #SI
kb = 1.38*10**(-23)    #SI



'''
Here we define 3 functions that we will use to make the animations and graphs. And also to read
the given data about the stellar model and have the ones we want in the format we are looking for.
'''

def animacion1D(F,x,t,k,titulo,guardar = g) -> None:
    '''
    Function that makes an animation of a function F(t,x)
    '''
    Nt = len(t)
    dt = t[1]-t[0]

    fig = plt.figure()
    fig.suptitle(titulo)
    ax = fig.add_subplot()
    ax.set_xlabel('r(m)')
    ax.set_ylabel('T(K)')
    ax.set_ylim((0,np.max(F[0,:])))
    line,= ax.plot(x ,F[0,:], label = 't = {} años'.format(0))
    L=ax.legend()
    def plotsim(i):
        M = 10**10
        if np.max(F[k*i,:]) < M/10:
            M = M/10
            ax.set_ylim((0,M))
        line.set_data(x,F[i*k,:])
        L.get_texts()[0].set_text('$t = {} años$'.format(round(t[i*k]/31.54,1),6))
        
        return line
    
    ani = animation.FuncAnimation(fig,plotsim,frames = Nt//k,interval = 1)
    plt.show()
    if guardar:
        ani.save('Temperatura.gif')


def plot(x,y,titulo,xlab,ylab, ylog = False,xlog = False,label = False,style = False) -> None:
    '''
    Function that makes a plot of y(x) with the given parameters
    '''
    plt.figure(titulo)
    plt.title(titulo)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if ylog:
        plt.semilogy()
    if xlog:
        plt.semilogx()
    if type(label) == str and type(style) == str:
        plt.plot(x,y,'.', label = label, linestyle = style)

    if type(label) == str and type(style) == bool:
        plt.plot(x,y,'.', label = label)

    if type(label) == bool and type(style) == str:
        plt.plot(x,y,'.', linestyle = style)

    if type(label) == bool and type(style) == bool:
        plt.plot(x,y,'.')
        

def leer(archivo) -> np.ndarray:
    '''
    Read data from a file column by column. 
    '''

    datos = []
    n = 0

    entrada = open(archivo,'r')
    lineas = [i for i in entrada]

    for i in lineas:
        datos.append([])
        for j in i.split():
            datos[n].append(float(j))
        n += 1
    entrada.close()
    datos = np.array(datos).transpose()
    return datos


def modelo(archivo, kr = 1) -> np.ndarray:
    '''
    Function that returns the data we want from our model
    '''

    datos = leer(archivo)
    
    r = datos[0][0:-kr]/100000                                             #km
    nr = len(r)

    rho = datos[1][0:-kr]                                                   #(densidad) g/cm^3
    nb = rho/(mn*1000)                                                      #(densidad de bariones)cm^(-3)

    ne = datos[2][0:-kr]*datos[1][0:-kr]/(mn*1000)                          #(densidad de electrones)cm^(-3)
    ye = ne/nb

    nn = datos[6][0:-kr]*datos[1][0:-kr]/(mn*1000)                          #(densidad de neutrones)cm^(-3)
    yn = nn/nb

    Np = (np.ones(len(r))-datos[6][0:-kr])*datos[1][0:-kr]/(mn*1000)        #(densidad de protones)cm^(-3)
    yp = Np/nb

    A =  datos[3][0:-kr]                                                    #número másico
    Z =  datos[4][0:-kr]                                                    #número atómico
    X =  datos[5][0:-kr]                                                    #fracción de átomos
 
    return r,rho,nb,ye,ne,nn,yn,Np,yp,A,Z,X


print('Close the plot to start the simulation.')

'''
Now we extract the data from the table and make the graphs of some important parameters
'''

r,rho,nb,ye,ne,nn,yn,Np,yp,A,Z,X = modelo(archivo)

plot(r,ye,'Fraction of electrons','r(km)','$y_e$',ylog=True)

plot(r,rho, 'Density in function of radius','r(km)','$\\rho$',ylog=True)

plot(r,A,'Nucleons in function of radius','r(km)','$A$')

plot(r,Z,'Protons in function of radius','r(km)','$Z$')

plot(r,X,'Fraction of nuclei in function of radius','r(km)','$X$')

plt.show()
print('Calculating...')




'''
Now we define the functions that we will use to calculate the physical magnitudes that we will
need in the resolution of the problem. The specific heat, the conductivity and the emissivity both
as a function of position and as a function of temperature. In addition to some auxiliary functions
such as the initial conditions of the star, one that we use for the outer boundary condition and
another to calculate the ratio between Coulomb energy and thermal energy.
'''

def Cv(T ,den = rho, YE=ye , AH = A , ZH = Z, XH = X, YN=yn,YP=yp) -> np.ndarray:         #erg*K^(-1)*cm^(-3)  
    '''
    Esta función calcula el Calor espercífico en función de la temperatura
    '''
    cvv =  np.array([cvtot(T[i]*normT,den[i],YE[i],AH[i],ZH[i],XH[i],YN[i],YP[i])[0] for i in range(len(r))])

    return cvv/normcv

    
def K(T, r, NE = ne) -> np.ndarray:                            #erg*K^(-1)*s^(-1)*cm^(-1) 
    '''
    Función que calcula la conductividad en función de la temperatura y la posición
    '''

    Kfe = (3*(np.pi**2)*ne)**(1/3)     #cm^(-1)
    Ke = 100*20.8*c*(Kfe)**2           #s^(-1)*cm^(-1) 

    return T*(Ke/(10**19))/normK


def emit(T,nbarion = nb, YE=ye , AH = A , ZH = Z, XH = X, YN=yn,YP=yp) -> np.ndarray:
    
    Epsilon = []
    
    for i in range(len(r)):
        Epsilon.append(QTOTAL(T[i]*normT,nbarion[i]/(10**39),YE[i] , AH[i]  , ZH[i] , XH[i] , YN[i],YP[i]))

    return -np.array(Epsilon)/normeps


def T0(r) -> np.ndarray:                                       #K
    '''
    Función para la condición inicial en la temperatura
    '''
    return (np.ones(len(r))*(10**10))/normT


def flujo(x,a,b) -> np.ndarray: 
    '''
    Función auxiliar para el cálculo de las condiciones de contorno
    '''
    return a*x**b


def gamma(T,r,rc,ZH = Z,rhog = rho,AH = A) -> np.ndarray: 
    '''
    Funicón que calcula el cociente entre la energía de coulomb y la térmica
    '''
    ZH = ZH[r > rc]
    AH = AH[r > rc]
    rhog = rhog[r > rc]
    return ((0.23*(ZH**2))/(T/(10**6)))*((rhog/AH)**(1/3))



'''
Now we define the function that will make the star evolve in time.
We will use the Crank-Nicholson method to solve the system of equations that describes
the cooling of the star, which is a parabolic PDE. We will also implement an adaptive
time step to ensure the stability of the method and to speed up the simulation when possible.
'''



def T_evo(x,dt_i,nt,ini): 
    '''
    Se le pasa a la función el array con los diferentes radios de los que 
    queremos calcular la temperatura, el primer paso de tiempo, el número de
    pasos de tiempo y el array con las temperaturas de los puntos que le hemos 
    pasado antes en el estado inicial. La función creará la matriz que representa
    el enfriamiento de la estrella contando la conducción y la emisión de la
    estrella de neutrones, calculando la temperatura en el instante siguiente.

    The function receives as parameters the array with the different radii of which we
    want to calculate the temperature, the first time step, the number of time steps and the
    array with the temperatures of the points that we have passed before in the initial state.
    The function will create the matrix that represents the cooling of the star counting conduction
    and emission of the neutron star, calculating the temperature at the next instant.
    '''
    
    T=np.zeros([nt,len(x)])      #We define the matrix that will contain the temperature
    t=[0]                        #We also define the array that will contain the time
    dt=[]                        #And the array that will contain the time steps
    
    k=1                          #This is a variable for controling the adaptative time step
    cteflujo1=3.5*10**(-15)      #Constant that appears in the linearization of the outer boundary condition
    cteflujo2=2.
    
    dx = x[1:]-x[:-1]            #We define the array with the different dx, since we are working with a non-uniform grid
    Nx=len(x)                    #We also define the number of points in the grid
    

    I=np.eye(Nx)                 
    
    diag = np.zeros(Nx)
    up,down = np.zeros(Nx-1),np.zeros(Nx-1)

    '''
    Now we start a loop with the index i, which in each iteration will calculate
    the temperature at instant i+1
    '''

    for i in range(nt-1):

        CV_t=Cv(T[i])               #We star the loop calculating the specific heat of all points for the previous instant
        K_t=K(T[i],x)               #The same with the thermal conductivity

        p = abs((np.sum(T[i][100:]*dx[99:])-np.sum(T[i-1][100:]*dx[99:]))/(np.sum(T[i][100:]*dx[99:])))
        
        '''
        Now we pass the function through a series of ifs to increase or decrease the time step
        '''
        
        if i == 0:                 #First step
            k = k

        elif p < 0.0002 :          #When the change in temperature is very small, we can increase the time step to speed up the simulation
            k=1.05*k  

        elif p > 0.0005 and k > 1: #When the change in temperature is bigger, we decrease the time step to ensure the stability of the method
            k=k/1.1

    
        t.append(t[i]+k*dt_i)      
        dt.append(t[-1]-t[-2])



        G=1-emit(T[i])*dt[i]*8/T[i]/CV_t   #Linalization of the term that depends on T in the emission
        R = np.ones(Nx)
        R[1:]=(dt[i])/(dx**2*CV_t[1:]*(x[1:]**2)*8)  #Some constants
    

        Q=1-cteflujo1*cteflujo2/T[i][-1] #Linearization of the term that depends on T in the outer boundary condition



        alfa = -(K_t[1:-1:]+K_t[:-2:])*((x[1:-1:]+x[:-2:])**2)  #Some magnitudes that are relative to the interfaces
        beta = -(K_t[1:-1]+K_t[2:])*((x[1:-1:]+x[2:])**2)       


        diag[1:-1:] = (1+R[1:-1:]*(alfa+beta)/G[1:-1:])         #Matrix elements
        up[1:] = -beta*R[1:-1:]/G[1:-1:]
        down[:-1:]=-alfa*R[1:-1:]/G[1:-1:]


        beta1=-(x[-2]*K_t[-2]+x[-1]*K_t[-1]+x[-2]*K_t[-1]+x[-1]*K_t[-2]) #Last row element

        diag[-1] =(1+R[-1]/G[-1]*beta1)/Q
        down[-1]=-beta1*R[-1]/G[-1]/Q       

        
        A = np.diag(diag,0) + np.diag(up,1) + np.diag(down,-1) #Matrix generation

        b = dt[i]*emit(T[i])/G/CV_t   #Defining the vector of independent terms, which depends on the emission and the specific heat
        b[-1]=b[-1]/Q                 #We also have to modify the last term of b to take into account the outer boundary condition
        b[0] = 0
        
        
        B = I-A/2                     #We aplicate the Crank-Nicholson method
        C = A/2


        C[0,0],C[0,1] = 0,0           #We also aply the BC to the matrices

        B[0,0],B[0,1] = 1,-1
        
        

        T[i+1]= np.linalg.solve(B,np.dot(C,T[i])+b)    #We solve the system to aplt Crank-Nicholson and get the temperature at the next instant

        T[i+1][-1]+=(4*cteflujo1*dx[-1]*R[-1]*(1-cteflujo2)*T[i][-1]**cteflujo2)/(G[-1]*Q)  #The last is aply de outer boundary condition to the temperature at the next instant

        
    return np.array(t),T


'''
We define the number of steps we want to take and the initial time step we are looking for
and we make the star evolve
'''


nt = 6000                               
dt = 70000/normt                        #s/normt


x=r*100000/normr                        #cm/normr


t1 = time.time()

Tiempo, Temperatura=T_evo(x,dt,nt,T0(r))

t2 = time.time()

print('Runtime:',round(t2-t1,2),'s')

print('Time simulated:',round(Tiempo[-1]/31.54,0), 'years')

animacion1D((Temperatura)*normT,r,Tiempo,5,'Cooling of a neutron star')

plt.figure()
plt.title('Temperature profile at different times')
plt.ylim((3*10**7,10**11))
plt.plot(r,Temperatura[0]*normT,label = 't = 0')
plt.plot(r,Temperatura[50]*normT,label = 't = {} days'.format(round(12*30.41*Tiempo[50]/31.54,2)))
plt.plot(r,Temperatura[500]*normT,label = 't = {} months'.format(round(2*Tiempo[500]/31.54,2)))
plt.plot(r,Temperatura[-1]*normT,label = 't = {} years'.format(round(Tiempo[-1]/31.54,2)))
plt.xlabel('r(km)')
plt.ylabel('T(K)')
plt.legend()
plt.semilogy()
plt.show()



'''
Now we study the state of the crust for certain times to see how it evolves
'''



rcrust = r[63]

GAM0 = gamma(Temperatura[0][r > rcrust]*normT,r, rcrust)
GAM50 = gamma(Temperatura[50][r > rcrust]*normT,r, rcrust)
GAM500 = gamma(Temperatura[500][r > rcrust]*normT,r, rcrust)
GAM2000 = gamma(Temperatura[2000][r > rcrust]*normT,r, rcrust)



plt.figure()
plt.title('$\Gamma$ profile at different times')
plt.plot(r[r> rcrust],GAM0,label = 't = 0')
plt.plot(r[r> rcrust],GAM50,label = 't = {} days'.format(round(12*30.41*Tiempo[50]/31.54,2)))
plt.plot(r[r> rcrust],GAM500,label = 't = {} months'.format(round(2*Tiempo[500]/31.54,2)))
plt.plot(r[r> rcrust],GAM2000,label = 't = {} years'.format(round(Tiempo[2000]/31.54,2)))
plt.plot(r[r> rcrust], 175*np.ones(len(r[r> rcrust])),'-.',label = '$\Gamma = 175$',)
plt.xlabel('r(km)')
plt.ylabel('$\Gamma$')
plt.legend()
plt.semilogy()
plt.show()


