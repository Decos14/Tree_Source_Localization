import numpy as np
import scipy as sp

def PositiveNormalMGF(t, mu, sigma2):
    return (np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2)))*2*(1-sp.stats.norm.cdf(np.sqrt(sigma2)*t))

def PositiveNormalMGFDerivative(t, mu, sigma2):
    intermediate_value_1 =  (np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2)))*(-1*mu+(sigma2)*t)*2*(1-sp.stats.norm.cdf(np.sqrt(sigma2)*t))
    intermediate_value_2 = np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2))*(np.sqrt(sigma2))*2*np.exp(-1*mu*(np.sqrt(sigma2)*t))*np.exp((1/2)*(sigma2)*((np.sqrt(sigma2)*t)**2))
    return intermediate_value_1 + intermediate_value_2

def PositiveNormalMGFDerivative2(t, mu, sigma2):
    intermediate_value_1 = (np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2)))*(mu**2 - 2*mu*sigma2*t+(sigma2**2)*(t**2)*sigma2)*2*(1-sp.stats.norm.cdf(np.sqrt(sigma2)*t))
    intermediate_value_2 = np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2))*(-1*mu+(sigma2)*t)*(np.sqrt(sigma2))*2*np.exp(-1*mu*(np.sqrt(sigma2)*t))*np.exp((1/2)*(sigma2)*((np.sqrt(sigma2)*t)**2))
    intermediate_value_3 = np.exp(-1*mu*t)*np.exp((1/2)*(sigma2)*(t**2))**(np.sqrt(sigma2))*2*np.exp(-1*mu*(np.sqrt(sigma2)*t))*np.exp((1/2)*(sigma2)*((-1*np.sqrt(sigma2)*t)**2))*(np.power(sigma2,3/2)*t+mu)
    return intermediate_value_1+intermediate_value_2+intermediate_value_3

def ExponentialMGF(t, lam):
    return lam/(lam+t)

def ExponentialMGFDerivative(t, lam):
    return lam/((lam+t)**2)

def ExponentialMGFDerivative2(t, lam):
    return -2*lam/((lam+t)**3)

def UniformMGF(t, start, stop):
    return (np.exp(-1*t*stop)-np.exp(-1*t*start))/(-1*t*(stop-start))

def UniformMGFDerivative(t, start, stop):
    return (start*np.exp(-1*t*start)-stop*np.exp(-1*t*stop))/(-1*t*(stop-start))

def UniformMGFDerivative2(t, start, stop):
    ((stop**2)*np.exp(-1*t*stop)-(start**2)*np.exp(-1*t*start))/(-1*t*(stop-start))

def PoissonMGF(t, lam):
    return np.exp(lam*(np.exp(-1*t)-1))

def PoissonMGFDerivative(t, lam):
    return -1*lam*np.exp(lam*(np.exp(-1*t)-1)-t)

def PoissonMGFDerivative2(t, lam):
    return (lam+np.exp(t))*lam*np.exp(lam*(np.exp(-1*t)-1)-2*t)

def AbsoluteCauchyMGF(t, sigma2):
    sigma = np.sqrt(sigma2)
    return (1/np.pi)*(2*sp.special.sici(t*sigma)[1]*np.sin(t*sigma)+np.cos(t*sigma)*(np.pi-2*sp.special.sici(t*sigma)[0]))

def AbsoluteCauchyMGFDerivative(t, sigma2):
    sigma = np.sqrt(sigma2)
    return (sigma/np.pi)*(2*np.cos(t*sigma)*sp.special.sici(t*sigma)[1]-np.sin(t*sigma)*(np.pi-2*sp.special.sici(t*sigma)[0]))

def AbsoluteCauchyMGFDerivative2(t, sigma2):
    sigma = np.sqrt(sigma2)
    return (sigma/(t*np.pi))*(2-2*t*sigma*sp.special.sici(t*sigma)[1]*np.sin(t*sigma)-t*sigma*np.cos(t*sigma)*(np.pi-2*sp.special.sici(sigma*t)[0]))