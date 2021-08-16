import numpy as np
from scipy import integrate
from scipy.special import zeta
from scipy import interpolate
from numba import jit

@jit
def engy(z, y):
    """Returns the dimensionless energy for a given z and dimensionless momentum y"""
    return np.sqrt(z * z + y * y)


@jit
def feq(z, y):
    """Returns a Maxwell-Boltzmann distribution of a particle in thermal equilibrium for a given z and momentum y"""
    return np.exp(-engy(z, y))


@jit
def funN(z, fN, yn, K):
    """Returns the RHS expression in eq 3.25"""
    en = engy(z, yn)
    return ((z * z * K) / en) * (np.exp(-en) - fN)


def fNyn(z_span, y0, z_eval, yn, K):
    """Integrates eq 3.25 over z for a given value of yn and K"""
    res = integrate.solve_ivp(funN, z_span, y0, t_eval=z_eval, args=(yn, K,), max_step= 1 / 200, method="RK23",
                              dense_output=True)
    return res.sol


def Neq(z, y):
    """Function returns N equilibrium"""
    feqarray = feq(z, y)  # Retrive array for feq
    integrand = np.multiply(feqarray, y * y / (4. * np.pi * np.pi * zeta(3))) # Integrand to integrate over yn
    soleq = integrate.simpson(integrand, axis=0)
    return soleq


def integrand1(yn, yl, z, Nl, fN, eps):
    """Returns the expression to be integrated over in eq 3.27"""
    if yl < (z * z - 4. * yl * yl) / (4. * yl):
        return 0.0
    else:
        feq = np.exp(-engy(z, yn))
        fun = fN(yn)
        return (yn / engy(z, yn)) * Nl * feq

def integrand2(yn, yl, z, Nl, fN, eps):
    """Returns the expression to be integrated over in eq 3.27"""
    if yl < (z * z - 4. * yl * yl) / (4. * yl):
        return 0.0
    else:
        feq = np.exp(-engy(z, yn))
        fun = fN(yn)
        return (yn / engy(z, yn)) * 2. * eps * (fun - feq)


def ylynintegral(z, Nl, fN, eps):
    """Performs the yl integral in eq 3.27 for a given z"""
    integral1 = integrate.nquad(integrand1, [[0., np.inf], [0., np.inf]],
        args  = (z, Nl, fN, eps,),
        opts  = {'epsrel': 1e-9, 'epsabs':1e-20},
        full_output = 1)
    integral2 = integrate.nquad(integrand2, [[0., np.inf], [0., np.inf]],
        args  = (z, Nl, fN, eps,),
        opts  = {'epsrel': 1e-9, 'epsabs':1e-20},
        full_output = 1)
    return integral1[0] - integral2[0]


def funL(z, Nl, fN, yn_vals, K, eps):
    """Returns the rhs of eq 3.27"""
    fNarr = fN(z)
    fN = interpolate.interp1d(yn_vals, fNarr, fill_value=(0, 0), kind='cubic', bounds_error=False)
    int = ylynintegral(z, Nl, fN, eps)
    return -z * z * K * int / 4.


def D2(z_span, z_eval, yn_vals, y, y0, K, eps):
    """Integrates the solution to eq 3.25 (fn) over yn to return the RHN number density, then Integrates eq 3.27 (Nl-l)
    over z"""
    functionN = fNyn(z_span, y0, z_eval, yn_vals, K)  # Returns a 2D array of fn evaluated for each z and yn
    integrand = np.multiply(functionN(z_eval), (y * y) / (4. * np.pi * np.pi * zeta(3)))  # fn*yn^2
    solN = integrate.simpson(integrand, axis=0)  # Integrates fn over yn using the simpson rule
    solL = integrate.solve_ivp(funL, z_span, [0], t_eval=z_eval, args=(functionN, yn_vals, K, eps,),
                               method="RK23", dense_output=True) # Integrates Nl-l over z
    solLres = solL.sol(z_eval)[0]
    return solN, np.abs(solLres)


eps = 1e-6 # Epsilon, amount of CP violation
K = [0.1, 10.]  # Decay parameter
z_span = [0.1, 50.]  # Upper and lower bounds of z
num_vals = 100

z_eval = np.logspace(-1., np.log10(50.), 100) # z points to evaluate at
np.savetxt('z_eval.txt', z_eval, delimiter=',')#    ax[0, k].loglog(z_eval, D2N, color='g', label='D2')

yn_vals = np.logspace(-3., 3., num_vals)  # Values of yn, I changed this to a log array as the fn tends to
                                                    # zero after about yn=10
y0 = np.zeros(num_vals)  # Initial value of differential equation


#Evaluate and plot D2, plots in figure 2
#fig, ax = plt.subplots(2, 2)
z, y = np.meshgrid(z_eval, yn_vals)
funeq = Neq(z, y)

for k in range(2):
    D2N, D2L = D2(z_span, z_eval, yn_vals, y, y0, K[k], eps)
    D2N = np.asarray(D2N) # Had to add this in because the function was returning the array in the form of a tuple and
    D2N = D2N.ravel() # I'm not sure why because I'm didn't not this before and I don't need to do it for Neq
    np.savetxt('funeq.txt', funeq, delimiter=',')#    ax[0, k].loglog(z_eval, D2N, color='g', label='D2')
    np.savetxt('D2L.txt', D2L, delimiter=',')#    ax[0, k].loglog(z_eval, D2N, color='g', label='D2')
    np.savetxt('zeval.txt', z_eval, delimiter=',')#    ax[0, k].loglog(z_eval, D2N, color='g', label='D2')
