import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import animation
import numpy.polynomial.polynomial as poly
import scipy.sparse
import scipy.sparse.linalg
from Emisividad import QTOTAL
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import newton

'''
Cooling of White Dwarfs
Yeray Antón and Marco Mas
'''

'''
First we define more physical constants for all the calculations
'''

mn = 1.675*10**(-27)   #SI
mp = 1.673*10**(-27)   #SI
me = 9.11*10**(-31)    #SI
n0 = 0.16*10**(39)     
hb = 1.054*10**(-34)   #SI
c = 3*10**8            #SI
kb = 1.38*10**(-23)    #SI

kbcgs = 1.38*10**(-16)


'''
Then we define the function that gives us the density profile of the star
'''

def radio_rho():
    '''
    Function that gives us the density profile of the star,
    solving the TOV equations for a polytropic equation of state with gamma = 4/3
    '''


    '''
    First we define some more relevant constants for this specific function
    '''
    G = 6.67430e-11             # Universal gravitational constant
    h = 6.63*1e-34
    u = 1.66054e-27             # Mass of 1 amu
    mu = 2                      # Mean molecular weight (dimensionless)
    K = h*c/8 * (3/np.pi)**(1/3)*(mu* u)**(-4/3)   # Polytrope K

    '''
    Initial conditions
    '''

    rho_0 = 1.2*1e11            # Initial density in Kg/m^3
    r0 = 10                     # Initial radius in m
    p0 = K*rho_0**(4/3)         # Initial pressure in Pa
    pf = p0*1e-13
    M0 = 4*np.pi*r0**3*rho_0/3  # Initial mass in Kg
    x0, xf = np.log(p0), np.log(pf)


    def rho(x):
        
        rho = (np.exp(x)/K)**(3/4)

        return rho


    def system(x, y):
        r, m = y
        if r <= 0:
            return [0, 0]
        
        dr_dx = - (r * (r - 2 * G * m / c**2) * np.exp(x)) /(G * (rho(x) + np.exp(x) / c**2) * (m + 4 * np.pi * r**3 * np.exp(x) / c**2))
        dm_dx = - (r**3 * 4 * np.pi * rho(x)*(r - 2 * G * m / c**2) * np.exp(x)) /(G * (rho(x) + np.exp(x) / c**2) * (m + 4 * np.pi * r**3 * np.exp(x) / c**2))

        return [dr_dx, dm_dx]

    '''
    Now we solve the system of equations using solve_ivp, and extract the solutions for r and M as functions of x=log(p)
    '''

    sol = solve_ivp(system, [x0, xf], [r0, M0], t_eval=np.linspace(x0, xf, 100))
    
    x_values = sol.t
    r_values = sol.y[0]
    M_values = sol.y[1]
    p_values = np.exp(x_values)

    rho_values=rho(x_values)

    return r_values,rho_values


print('You want to save the animation (It may take 5 minutes):')
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
    ax.set_yscale('log')
    ax.set_ylim((10**5,5*np.max(F[0,:])))
    line,= ax.plot(x ,F[0,:], label = 't = {} years'.format(0))
    L=ax.legend()
    def plotsim(i):
        line.set_data(x,F[i*k,:])
        L.get_texts()[0].set_text('$t = {} years$'.format(round(t[i*k]/31.54,1),6))
        
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
        
        
def modelo() -> np.ndarray:
    '''
    Function that gives us the different parameters of the stellar model as functions of the radius,
    like the density, the number of baryons, the fraction of electrons, etc.
    '''

    r,rho = radio_rho()

    
    r = r/1000                                                #km
    nr = len(r)

    rho = rho/1000                                            #(density) g/cm^3
    nb = rho/(mn*1000)                                        #(baryon density) cm^(-3)

    ye = 0.5*np.ones(nr)
    ne = ye*nb                                                #(electron density) cm^(-3)
    
    yn = 0.5*np.ones(nr)
    nn = yn*nb                                                #(neutron density) cm^(-3)

    yp = 0.5*np.ones(nr)
    Np = yp*nb                                                #(proton density) cm^(-3)


    A =  12*np.ones(nr)                                       #mass number
    Z =  6*np.ones(nr)                                        #atomic number
    X =  np.ones(nr)                                          #atomic fraction
 
    return r,rho,nb,ye,ne,nn,yn,Np,yp,A,Z,X


'''
Now we and make the graphs of some important parameters
'''

r,rho,nb,ye,ne,nn,yn,Np,yp,A,Z,X = modelo()

plot(r,ye,'Proton fractions','r(km)','$y_e$',ylog=True)

plot(r,rho,'Density vs Radius','r(km)','$\\rho$',ylog=True)

plot(r,A,'Nucleons vs Radius','r(km)','$A$')

plot(r,Z,'Protons vs Radius','r(km)','$Z$')

plot(r,X,'Fraction of nuclei vs Radius','r(km)','$X$')

plt.show()

print('Close the graphs to start the simulation.')

print('Calculating...')




'''
Now, we define the functions that we will use to calculate the physical magnitudes 
that we will need in the resolution of the problem. The specific heat, the conductivity, 
and the emissivity both as a function of position and as a function of temperature.
In addition to some auxiliary functions such as the initial conditions of the star, one that we use 
for the outer boundary condition and another to calculate the ratio between Coulomb energy and thermal energy.
'''

def Cv(T ,den = rho,NE =ne ,YE=ye , AH = A , ZH = Z, XH = X, YN=yn,YP=yp) -> np.ndarray:         #erg*K^(-1)*cm^(-3)  
    '''
    This function calculates the Specific Heat as a function of temperature
    '''
    cvv =  3*kbcgs*ne/2

    return cvv/normcv

    
def K(T, r, NE = ne) -> np.ndarray:                            #erg*K^(-1)*s^(-1)*cm^(-1) 
    '''
    Function that calculates the conductivity as a function of temperature and position
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
    Function for the initial temperature condition
    '''
    return (np.ones(len(r))*(10**8))/normT


def flujo(x,a,b) -> np.ndarray: 
    '''
    Auxiliary function for the calculation of boundary conditions
    '''
    return a*x**b


def gamma(T,r,rc,ZH = Z,rhog = rho,AH = A) -> np.ndarray: 
    '''
    Function that calculates the ratio between the Coulomb energy and the thermal energy
    '''
    ZH = ZH
    AH = AH
    rhog = rhog
    return ((0.23*(ZH**2))/(T/(10**6)))*((rhog/AH)**(1/3))



plot(r,Cv(T0(r))*normcv,'Specific Heat','r(km)','$C_v$ $(erg/cm^3 \cdot K)$')

plot(r,K(T0(r),r)*normK,'Conductivity','r(km)','$K$ $(erg/s \cdot cm \cdot K)$')

plt.show()


'''
We implement the function that evolves temperature
'''


def T_evo(x,dt_i,nt,ini): 
    '''
    The function receives the array with the different radii of which we want 
    to calculate the temperature, the first time step, the number of time steps, 
    and the array with the temperatures of the points that we passed before in the initial state. 
    The function will create the matrix that represents the cooling of the star counting 
    conduction and emission of the neutron star, calculating the temperature at the next instant.
    '''
    
    T=np.zeros([nt,len(x)])      #We define the matrix of temperatures
    T[0]=ini                     #We make our matrix have the initial values parameter as the first element
    t=[0]                        #We create a time list, which we will complete
    dt=[]                        #The same but with time differentials
    
    k=1                          #We define this auxiliary magnitude to modulate the value of dt without harming the stability of the method
    cteflujo1=5.67*10**(-17)     #These two constants are for the outer boundary condition
    cteflujo2=4.
    
    dx = x[1:]-x[:-1]            #We define the array of radius increments to use it comfortably
    Nx=len(x)                    #We also define the size of our simulation cell
    

    I=np.eye(Nx)                 #We also define the identity matrix now to use it later
    
    diag = np.zeros(Nx)
    up,down = np.zeros(Nx-1),np.zeros(Nx-1)

    '''
    Now we start a loop with the index i, which in each iteration will calculate
    the temperature at instant i+1
    '''

    for i in range(nt-1):

        CV_t=Cv(T[i])               #The loop starts calculating the specific heat of all points for the previous instant
        K_t=K(T[i],x)               #We do the same for the conductivity

        p = abs((np.sum(T[i][100:]*dx[99:])-np.sum(T[i-1][100:]*dx[99:]))/(np.sum(T[i][100:]*dx[99:])))
        
        '''
        Now we pass the function through a series of ifs to increase or decrease the time step
        '''
        
        if i == 0:                 #For the first step
            k = k

        elif p < 0.0002 :          #For steps where we can increase the dt
            k=1.05*k  

        elif p > 0.0005 and k > 1: #For when we must reduce dt to maintain stability
            k=k/1.1

    
        t.append(t[i]+k*dt_i)      #We add t and dt respectively
        dt.append(t[-1]-t[-2])



        G=1-emit(T[i])*dt[i]*8/T[i]/CV_t   #Term derived from linearizing the emissivity
        R = np.ones(Nx)
        R[1:]=(dt[i])/(dx**2*CV_t[1:]*(x[1:]**2)*8)  #Grouped constants term
    

        Q=1-cteflujo1*cteflujo2/T[i][-1] #Term due to the linearization of the outer boundary condition



        alfa = -(K_t[1:-1:]+K_t[:-2:])*((x[1:-1:]+x[:-2:])**2)  #Auxiliary magnitudes corresponding
        beta = -(K_t[1:-1]+K_t[2:])*((x[1:-1:]+x[2:])**2)       #to the interfaces


        diag[1:-1:] = (1+R[1:-1:]*(alfa+beta)/G[1:-1:])         #We define the matrix elements
        up[1:] = -beta*R[1:-1:]/G[1:-1:]
        down[:-1:]=-alfa*R[1:-1:]/G[1:-1:]


        beta1=-(x[-2]*K_t[-2]+x[-1]*K_t[-1]+x[-2]*K_t[-1]+x[-1]*K_t[-2]) #Missing term in the last row

        diag[-1] =(1+R[-1]/G[-1]*beta1)/Q
        down[-1]=-beta1*R[-1]/G[-1]/Q       

        
        
        A = np.diag(diag,0) + np.diag(up,1) + np.diag(down,-1) #We create the matrix with the magnitudes defined previously

        b = dt[i]*emit(T[i])/G/CV_t   #We define the vector that does not depend linearly on T
        b[-1]=b[-1]/Q                 #We adjust this vector so it satisfies the boundary conditions
        b[0] = 0
        
        
        B = I-A/2                     #We apply Crank-Nicholson dividing the matrix in two
        C = A/2



        C[0,0],C[0,1] = 0,0           #We modify the matrices for the BC of point 0

        B[0,0],B[0,1] = 1,-1
        
        

        T[i+1]= np.linalg.solve(B,np.dot(C,T[i])+b)    #We multiply T[i] by the matrix and add it to b to get the independent components vector
                                                       #of the system, and use B as the coefficients matrix

        T[i+1][-1]+=(4*cteflujo1*dx[-1]*R[-1]*(1-cteflujo2)*T[i][-1]**cteflujo2)/(G[-1]*Q)  #Last adjustment of the last temperature to satisfy the outer BC

        
    return np.array(t),T


'''
We define the number of steps we want to take and the initial time step we are looking for
And we make the star evolve
'''


nt = 100000                               
dt = 5000000/normt                           #s/normt


x=r*100000/normr                        #cm/normr


t1 = time.time()

Tiempo, Temperatura=T_evo(x,dt,nt,T0(r))

t2 = time.time()

print('Runtime:',round(t2-t1,2),'s')

print('Total simulated time:',round(Tiempo[-1]/31.54,0), 'years')

animacion1D((Temperatura)*normT,r,Tiempo,50,'Time evolution of temperature')

plt.figure()
plt.title('Temperature vs Radius for various times')
plt.ylim((10**6,3*10**8))
plt.plot(r,Temperatura[-1]*normT,label = 't = {} years'.format(round(Tiempo[-1]/31.54,2)))
plt.plot(r,Temperatura[5000]*normT,label = 't = {} years'.format(round(Tiempo[5000]/31.54,2)))
plt.plot(r,Temperatura[500]*normT,label = 't = {} years'.format(round(Tiempo[500]/31.54,2)))
plt.plot(r,Temperatura[0]*normT,label = 't = 0')
plt.xlabel('r(km)')
plt.ylabel('T(K)')
plt.legend()
plt.semilogy()
plt.show()



'''
Now we study the state of the crust for certain times to see how it evolves
'''



rcrust = r[63]

GAM0 = gamma(Temperatura[0]*normT,r, rcrust)
GAM50 = gamma(Temperatura[500]*normT,r, rcrust)
GAM500 = gamma(Temperatura[5000]*normT,r, rcrust)
GAM2000 = gamma(Temperatura[-1]*normT,r, rcrust)



plt.figure()
plt.title('$\Gamma$ as a function of position')
plt.plot(r,GAM2000,label = 't = {} years'.format(round(Tiempo[-1]/31.54,2)))
plt.plot(r,GAM500,label = 't = {} years'.format(round(Tiempo[5000]/31.54,2)))
plt.plot(r,GAM50,label = 't = {} years'.format(round(Tiempo[500]/31.54,2)))
plt.plot(r,GAM0,label = 't = 0')
plt.plot(r, 175*np.ones(len(r)),'-.',label = '$\Gamma = 175$',)
plt.xlabel('r(km)')
plt.ylabel('$\Gamma$')
plt.legend()
plt.semilogy()
plt.show()