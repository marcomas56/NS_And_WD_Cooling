import numpy as np
import matplotlib.pyplot as plt


# Define the cvf function  (EVERYTHING in c.g.s UNITS)
def cvtot(t, rho, YE, AH, ZH, XH, YN, YP):
     
    mn = 1.66e-24
    nbaryon = 1.e-39 * rho / mn   # in fm^(-3)
    n_n = nbaryon * YN
    n_p = nbaryon * YP
    n_e = nbaryon * YE
    nions = nbaryon * XH / AH 
    
    effmn, effmp = eff_mass(n_n, n_p)  # Effective masses in units of the rest baryon mass
    
    cvn = ccvfn(t, n_n, effmn)
    cvp = ccvfp(t, n_p, effmp)
    cvion = ccvfion(t, nions, ZH, AH)
    cve = ccvfe(t, n_e)
    
    cvf = cvion + cve + cvn + cvp
    
    return [cvf, cvion, cve, cvn, cvp]

def eff_mass(n_n, n_p):
  
    # Calculate kn and kp
    kn = (3*np.pi**2 * n_n) ** (1.0 / 3.0)
    kp = (3*np.pi**2 * n_p) ** (1.0 / 3.0)

    # Fit for effmn
    a = [ 1.0, -1.43453622,  1.68252423, -0.93623544,  0.20018045]
    effmn = a[0] + a[1] * kn + a[2] * kn**2 + a[3] * kn**3 + a[4]*kn**4

    # Fit for effmp
    a = np.array([ 1.0,  -0.0240683,  -0.18472525,  0.05734536,  0.00600171])
    effmp = a[0] + a[1] * kp + a[2] * kp**2 + a[3] * kp**3 + a[4]*kp**4

    return effmn, effmp
  
def ccvfn(t, n_n, effmn):
#   Neutrons' specific heat
    msn = effmn*939.6/197.33  # neutron effective mass in fm^(-1)
    xn = (3*np.pi**2 * n_n) ** (1.0 / 3.0) / msn
    ccvfn = 4.5507e+11 * effmn**2 * xn * np.sqrt(xn**2 + 1.0) * t 


    return ccvfn

def ccvfp(t, n_p, effmp):
#   Protons' specific heat
    msn = effmp*938.3/197.33  # proton effective mass in fm^(-1)
    xp = (3*np.pi**2 * n_p) ** (1.0 / 3.0) / msn
    ccvfp = 4.5507e+11 * effmp**2 * xp * np.sqrt(xp**2 + 1.0) * t 

    
    return ccvfp

def ccvfe(t, ne):
    me = 0.511/197.33  # electron mass in fm^(-1)
    xe = (3*np.pi**2 * ne) ** (1.0 / 3.0) / me

    #ccvfe = 4.5507e+11 * me**2 * xe * np.sqrt(xe**2 + 1.0) * t 
    ccvfe = ne * 2.298976e14 * t * np.sqrt(xe**2 + 1.0) / xe**2
    return ccvfe

def ccvfion(t, nions, ZH, AH):
    kb = 8.6e-11  # Mev/K
    alpha = 1./137.
    hbarc = 197.33
    mion = AH*938/197.33  # ion mass in fm^(-1)
    
    tplasma = hbarc*np.sqrt(4*np.pi*alpha*ZH*nions/mion) / kb
    tdebye = 0.45*tplasma  # check the origin of this approximation
    
    xi = (4*np.pi*nions/3.)**(1./3.) * hbarc # 1/r_i in MeV
    gamma = alpha * ZH**2 * xi / (kb*t)

    kbcgs = 1.38e-16*1e39  # c.g.s. to fm^3
    if gamma <= 1.0:  
        ccvfion = kbcgs * nions * 3.0 / 2.0
    elif gamma >= 150.0:
        ccvfion = kbcgs * nions * 3.0 * fdebye(t / tdebye)
    else:
        ccvfion = kbcgs * nions * 3.0 * (1.0 + np.log10(gamma) / np.log10(150.0)) / 2

    return ccvfion

def fdebye(x):
    if x <= 0.15:
        fdebye = 77.9273 * x**3
    elif x >= 0.4:
        fdebye = 1.0 - 1.0 / (20.0 * x**2)
    else:
        fdebye = 1.69798 * x + 0.0083073
    return fdebye




# Define array of temperature values
t_values = np.linspace(1.e6, 1.e8, 100)  # Example temperature range from 0 to 10

# Choose fixed values for other parameters
rho_values = np.logspace(6, 15, 100)
YE = 0.1
AH = 56.0
ZH = 28.0
XH = 1.0
YN = 0.4
YP = 0.1

# Call cvf for each temperature value
t = 1.e9

results  = np.array([cvtot(t, rho, YE, AH, ZH, XH, YN, YP) for rho in rho_values])



# Plot results
plt.figure(figsize=(8, 6))
plt.plot(rho_values, results[:,0], label=r'$C_v$')
plt.plot(rho_values, results[:,1], label=r'$C_i$')
plt.plot(rho_values, results[:,2], label=r'$C_e$')
plt.plot(rho_values, results[:,3], label=r'$C_n$')
plt.plot(rho_values, results[:,4], label=r'$C_p$')
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'$\rho$')
plt.ylabel('cvf')
plt.title('Specific heat')
plt.grid(True)
plt.legend()
plt.show()
