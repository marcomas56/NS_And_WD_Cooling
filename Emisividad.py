import numpy as np
import matplotlib.pyplot as plt

def QTOTAL(T, nbaryon, YE, AH, ZH, XH, YN, YP):   # T IN K , NBARYON IN FM^(-3)
  
    t9 = 1e-9 * T
    
    nbar0 = nbaryon/0.153   # in units of saturation density

    fnn = nbaryon * YN
    fnp = nbaryon * YP
    effmn, effmp = eff_mass(fnn, fnp)

    qmur = Qmurca(t9, nbar0, effmn, effmp, YE, YN, YP)
    qdu = Qdurca(t9, effmn, effmp, XH, YE, YP)
    qnn = QBnn(t9, nbar0, effmn, YN, XH)
    qnp = QBnp(t9, nbar0, effmn, effmp, YP)
    qpp = QBpp(t9, nbar0, effmp, YP)
    qep = QepBREMS(t9, nbar0, effmp, YP)
    qeA = QeABREMS(t9, nbar0, XH)
    qpl = QPLASMA(t9, nbar0, YE, XH)
    qpair = QPAIR(t9, nbar0, ZH, AH)

    qtot = qmur + qdu + qnn + qnp + qpp + qep + qeA + qpl + qpair

    return qtot

def QBnn(T9, n0, EFFMN, YN, XH):
    e3 = 1.0 / 3.0
    alphann = 0.59
    fmu = 1.0
    if 0.0 < XH < 1.0:
        fmu = YN
        kFn = (3 * np.pi**2 * n0 * YN * 0.156)**(1./3.)  # kFermi in fm^-1
        kFn *= 197.326
        xmpi = 139.0
        xu = xmpi / (2.0 * kFn)
        alphann = 1.0 - 3.0 * xu * np.arctan(1.0 / xu) / 2.0 + xu**2 / (2.0 * (1.0 + xu**2))
    if XH == 1.0:
        alphann = 0.0
        
    betann = 0.56
    nfnu = 3.0
    RBnn = 1.0
    QBnn = 7.4e19 * EFFMN**4 * (n0*YN)**e3 * alphann * betann * 3 * T9**8 * nfnu * fmu
    return QBnn

def QBnp(T9, n0, EFFMN, EFFMP, YP):
    alphanp = 1.06
    betanp = 0.66
    nfnu = 3.0
    e3 = 1.0 / 3.0
    RBnp = 1.0
    QBnp = 1.5e20 * (EFFMN * EFFMP)**2 * (n0*YP)**e3 * alphanp * betanp * nfnu * T9**8
    return QBnp
5
def QBpp(T9, n0, EFFMP, YP):
    e3 = 1.0 / 3.0
    alphapp = 0.11
    betapp = 0.7
    nfnu = 3.0
    RBpp = 1.0
    QBpp = 7.4e19 * EFFMP**4 * (n0*YP)**e3 * alphapp * betapp * 3 * nfnu * T9**8
    return QBpp

def Qmurca(T9, n0, EFFMN, EFFMP, YE, YN, YP):
    """
     CALCULATES MODIFIED URCA NEUTRINO EMISSIVITY 
     FRIMAN AND MAXWELL (1979) Ap. J. 232, 541  --------------------- (FM79)
     Yakovlev & Levenfish (1995) A&A, 297, 717  --------------------- (YL95)
    """
    e3 = 1.0 / 3.0
    
    RMn = 1.0
    RMp = 1.0
    
    alphan = 1.76 - 0.63 / (n0*YN + 1e-20)**(2.0 / 3.0)
    betan = 0.68
    if alphan > 0.0 and YP > 0.0:
        qMn = 8.55e21 * EFFMN**3 * EFFMP * (n0*YE)**e3 * T9**8 * alphan * betan
    else:
        qMn = 0.0
    if alphan > 0.0 and YP > 0.0 and YE**e3 + 3.0 * YP**e3 > YN**e3:
        qMp = 8.53e21 * EFFMP**3 * EFFMN * (n0*YE)**e3 * T9**8 * alphan * betan * max((1.0 - 0.25 * (YE / YP)**e3), 0.0)
    else:
        qMp = 0.0
          
    return qMn + qMp

def Qdurca(T9, effmn, effmp, xh, ye, yp):
    '''
    C-------------------------------------------------------------------
    C----- CALCULATES DIRECT URCA NEUTRINO EMISSIVITY ------------------
    C----- Lattimer et al. (1991) Phys. Rev. Lett., 66, 2701  ----------
    C-------------------------------------------------------------------
    '''
    if xh > 0.0:  # only in the SF core
      return 0.
    
    if yp >= 0.11:
        qdu_e = 4.0e27 * effmn * effmp * T9**6 * yp**(1.0/3.0)
    else:
        qdu_e = 0.0

    return qdu_e
  
def QeABREMS(t9, n0, xh):
    """
    Calculates ν emissivity from e-N Bremsstrahlung.
    From Kaminker, Pethick et al., Astron. & Astrophys. (1999)343,1009
    Validity: 5d7 < t < 2d9 [K], 1d9 < rho < 1.4d14 [g/cm3]
    """
    if not (0.05 <= t9 <= 2 and 1e-5 <= n0 <= 0.5 and xh > 0):
        return 0.0

    xtau = np.log10(10*t9)
    xr = np.log10(280.*n0)

    a1, a2, a3, a4, a5, a6, a7, a8, a9 = 11.204, 7.304, 0.2976, 0.370, 0.188, 0.103, 0.0547, 6.77, 0.228
    QeABREMS = 10 ** (a1 + a2 * xtau + a3 * xr - a4 * xtau**2 + a5 * xtau * xr - a6 * xr**2 + a7 * xtau**2 * xr - a8 * np.log(1 + a9 * n0))
    
    return QeABREMS

def QPLASMA(t9, n0, ye, xh):
    """
    Calculates plasma neutrino emissivity.
    Yakovlev et al., Phys.Rep. 354 (2001) 1-155.
    """
    kB = 8.617e-5  # k Boltzmann in eV/K
    me = 0.510998e6  # me in eV/c^2
    Qc = 1.203e23
    xtr = kB * 1e9 * t9 / me
    
    kFe = (3 * np.pi**2 * n0 * ye * 0.156)**(1./3.)  # kFermi in fm^-1
    kFe *= 197.326e6  # kFe in eV
    xr = kFe / me
    alpha = 1 / 137.0
    fp = np.sqrt(4 * alpha * xr**3 / (3 * np.pi * np.sqrt(1 + xr**2))) / xtr
    zexp1 = -fp
    if zexp1 < -150:
        zexp1 = -150  # cutoff

    XIP = xtr**9 * (16.23 * fp**6 + 4.604 * fp**7.5) * np.exp(zexp1)
    sumCV2 = 0.9248
    if xh == 0.0:
        Qc = 0.0  # no plasma if core

    QPLASMA = Qc * XIP * sumCV2 / (96 * np.pi**4 * alpha)
    
    return QPLASMA

def QepBREMS(T9, n0, effmp, YP):
    """
    Calculates Bremsstrahlung electron-proton neutrino emissivity.
    Reference: Maxwell, 79, ApJ 231, 201
    """
    if YP > 0.0:
        QepBREMS = 2.4e17 * (n0*YP)**(-2/3) * T9**8 * effmp**2
    else:
        QepBREMS = 0.0
    
    return QepBREMS

def QPAIR(t9, n0, z, a):
    """
    Calculates pair neutrino emissivity.
    Itoh et al., (Feb, 1996) Ap. J. Suppl.
    """
    coef1 = 0.840766
    coef2 = 0.090766

    if t9 < 0.3:
        return 0.0

    x = t9 / 5.9302
    x2 = x * x
    xi = 1.0 / x
    za = z / a
    rza = 2.8e14 * n0 * za
    chi = 1e-3 * rza**(1/3) * xi

    if t9 < 10:
        b1, b2, b3, c = 0.9383, -0.4141, 0.05829, 5.5924
    else:
        b1, b2, b3, c = 1.2383, -0.8141, 0.0, 4.9924

    f = (6.002e19 + chi * (2.084e20 + chi * 1.872e21)) * np.exp(-c * chi) / (chi**3 + xi * (b1 + xi * (b2 + xi * b3)))
    g = 1.0 - x2 * (13.04 - x2 * (133.5 + x2 * (1534.0 + x2 * 918.6)))
    q = (1.0050 + 0.3967 * np.sqrt(x) + 10.7480 * x2)**(-1) * (1.0 + rza / (7.692e7 * x**3 + 9.715e6 * np.sqrt(x)))**(-0.3)
    QPAIR = (coef1 + coef2 * q) * g * np.exp(-2 * xi) * f
    
    return QPAIR 

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

'''
# Define the density range in log scale
density_range = np.logspace(-5, 0, num=20)
emissivity = np.zeros_like(density_range)

# Define other parameters
T = 1e10  # Example temperature
YE = 0.5  # Example electron fraction
AH = 1  # Example atomic mass number
ZH = 1  # Example atomic charge number
XH = 0.7  # Example hydrogen fraction
YN = 0.3  # Example neutron fraction
YP = 0.5  # Example proton fraction

# Loop over densities
i = 0
for nbar in density_range:
    # Call QTOTAL function for each density
    emissivity[i] = QTOTAL(T, nbar, YE, AH, ZH, XH, YN, YP)
    i+=1
    print ("{:10.3E}, {:10.3E}".format(nbar, emissivity[i-1]))
    #print(f"Density: {n_B}, Q_nu: {emissivity[i-1]}")
    
# Plot QTOTAL vs density
plt.figure(figsize=(10, 6))
plt.plot(density_range, emissivity, marker='o', linestyle='-')
plt.xscale("log")
plt.yscale("log")
plt.title('QTOTAL vs baryon density')
plt.xlabel(r'$n_B$ (fm$^{-3}$)')
plt.ylabel('QTOTAL (c.g.s)')
plt.grid(True)
plt.show()
'''