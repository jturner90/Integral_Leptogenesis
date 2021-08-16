import numpy as np
from scipy import integrate
from scipy import interpolate
from numba import jit
import vegas

myglob = 0

class Calculator(object):
    import numpy as np
    def __init__(self, K, yn, eps=1e-6, tMin=0.1, tMax=50):
        self.y0_    = np.zeros_like(yn)
        self.yn_    = yn#np.logspace(np.log10(tMin), np.log10(tMax), num_vals)
        self.dlogy  = np.log10(self.yn_[1]) - np.log10(self.yn_[0])
        self.dlogy0 = np.log10(self.yn_[0])
        self.tMin_  = tMin
        self.tMax_  = tMax
        self.K_     = K
        self.currz_ = None
        self.Y_     = None
        self.solve()

    def engy(self, z, y):
        return np.sqrt(z * z + y * y)


    def feq(self, z, y):
        return np.exp(-self.engy(z, y))


    def fun(self, z, fN):
        en = self.engy(z, self.yn_)
        return (z * z * self.K_) / en * (np.exp(-en) - fN)


    def solve(self, method="RK23", max_step=1/200.):
        from scipy import integrate
        self.sol_ = integrate.solve_ivp(self.fun, [self.tMin_, self.tMax_], self.y0_, max_step=max_step, method=method, dense_output=True)

    def logindex(self, y):
        import math
        return math.floor(  (math.log10(y) - self.dlogy0) / self.dlogy)

    def lininterp(self, x, x1, x2, y1, y2):
        return y1 + (x - x1)*(y2 - y1) / (x2 - x1)

    def eval(self, z, y):
        if z!=self.currz_:
            self.Y_ = self.sol_.sol(z)
            self.currz_ = z

        yindex = self.logindex(y)

        return self.lininterp(y, self.yn_[yindex], self.yn_[yindex+1], self.Y_[yindex], self.Y_[yindex+1])
@jit
def engy(z, y):
    return np.sqrt(z * z + y * y)

@jit
def feq(z, y):
    return np.exp(-engy(z, y))

def funN(z, fN, yn, K):
    en = engy(z, yn)
    return (z * z * K) / en * (np.exp(-en) - fN)

def fNyn(z_span, y0, z_eval, yn, K):
    res = integrate.solve_ivp(funN, z_span, y0, t_eval=z_eval, args=(yn, K,), max_step= 1 / 200, method="RK23", dense_output=True)
    return res.sol(z_eval)

def ynintegrand2A(yn, z, Nl, fN, eps):
    import math
    en      = math.sqrt(z * z + yn * yn)
    fun     = calc.eval(z, yn)
    fNeq    = math.exp(-en)
    delta   = (Nl * fNeq)
    return (yn / en) * delta
    
def ynintegrand2B(yn, z, Nl, fN, eps):
    import math
    en      = math.sqrt(z * z + yn * yn)
    fun     = calc.eval(z, yn)
    fNeq    = math.exp(-en)
    kappa   = fun - fNeq
    delta   = (2. * eps * (kappa))
    return (yn / en) * delta

def ynIntegrandVegas2A(x, z, Nl, fN, eps):
    yl, yn = x
    lowerlim = (z * z - 4. * yl * yl) / (4. * yl)
    if yl < lowerlim:
        return 0.0
    else:
        return ynintegrand2A(yn, z, Nl, fN, eps)

def ynIntegrandVegas2B(x, z, Nl, fN, eps):
    yl, yn = x
    lowerlim = (z * z - 4. * yl * yl) / (4. * yl)
    if yl < lowerlim:
        return 0.0
    else:
        return ynintegrand2B(yn, z, Nl, fN, eps)
        

def ylintegralA(z, Nl, fN, eps):
    integ(lambda x: ynIntegrandVegas2A(x, z, Nl[0], fN, eps), nitn=4, neval=1000)
    aha = integ(lambda x: ynIntegrandVegas2A(x, z, Nl[0], fN, eps), nitn=4, neval=10000)
    try:
        return aha[0].val
    except:
        return aha.val
        
def ylintegralB(z, Nl, fN, eps):
    integ(lambda x: ynIntegrandVegas2B(x, z, Nl[0], fN, eps), nitn=4, neval=1000)
    aha = integ(lambda x: ynIntegrandVegas2B(x, z, Nl[0], fN, eps), nitn=4, neval=10000)
    try:
        return aha[0].val
    except:
        return aha.val


def funL(z, Nl, fN, K, eps):
    print(z)
    int = ylintegralA(z, Nl, fN, eps) - ylintegralB(z, Nl, fN, eps)
    return (-z * z * K * 0.25) * int


def Neq(z, y):
    feqarray  = feq(z, y)  # Retrive array for feq
    integrand = np.multiply(feqarray, y * y / (np.pi * np.pi)) # Integrand to integrate over yn
    soleq     = integrate.simpson(integrand, axis=0)
    return soleq


def D2(z_span, z_eval, yn_vals, y, y0, K, eps):
    fNarray  = calc.sol_.sol(z_eval)  # Returns a 2D array of fn evaluated for each z and yn
    integrand = np.multiply(fNarray, (y * y) / (np.pi * np.pi))  # fn*yn^2
    solN      = integrate.simpson(integrand, axis=0)  # Integrates fn over yn using the simpson rule
    fN=None
    solL      = integrate.solve_ivp(funL, z_span, [0], t_eval=z_eval, args=(fN, K, eps,), dense_output=True, method="RK45")
    solLres   = solL.sol(z_eval)[0]
    return solN, np.abs(solLres)


eps = 1e-6
K = [0.1, 10.]
z_span = [0.1, 50.]  # Upper and lower bounds of z
num_vals = 1000

z_eval  = np.logspace(np.log10(z_span[0]), np.log10(z_span[1]), 100) # z points to evaluate at

yn_vals = np.logspace(-5, 3., num_vals)
y0 = np.zeros(num_vals)

z, y = np.meshgrid(z_eval, yn_vals)
funeq = Neq(z, y)
k=1
calc = Calculator(K[k], yn_vals, tMin=z_span[0], tMax=z_span[1])

print("Doing k={}".format(k))
integ = vegas.Integrator([[0, 1000], [0, 1000]])
D2N, D2L = D2(z_span, z_eval, yn_vals, y, y0, K[k], eps)
D2N      = np.asarray(D2N)
D2N      = D2N.ravel()
np.savetxt('z_eval.txt', z_eval, delimiter=',')
np.savetxt('funeq.txt', funeq, delimiter=',')
np.savetxt('D2L.txt', D2L, delimiter=',')
