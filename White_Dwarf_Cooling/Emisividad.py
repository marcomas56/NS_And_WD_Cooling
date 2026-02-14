import numpy as np
import matplotlib.pyplot as plt

def QTOTAL(T, nbaryon, YE, AH, ZH, XH, YN, YP):   # T IN K , NBARYON IN FM^(-3)
  
    t9 = 1e-9 * T
    
    nbar0 = nbaryon/0.153   # in units of saturation density

    fnn = nbaryon * YN
    fnp = nbaryon * YP


    qeA = QeABREMS(t9, nbar0, XH)


    qtot = qeA 

    return qtot

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
