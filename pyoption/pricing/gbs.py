"""
Generalized black scholes model for European options
"""

import numpy as np
import scipy
from scipy.stats import norm
import math


def _generalized_black_scholes(option_type="call", *, fs, x, t, r, b, v):
    """
    :param option_type: one of ('call', 'put')
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params b: cost of carry (in percentage)
    :params v: implied volatility (in percentage)
    """
    t_sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t_sqrt)
    d2 = d1 - v * t_sqrt

    if option_type == "call":
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        value = fs * math.exp((b - r) * t) * cdf_d1 - x * math.exp(-r * t) * cdf_d2
        delta = math.exp((b - r) * t) * cdf_d1
        gamma = math.exp((b - r) * t) * pdf_d1 / (fs * v * t_sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * pdf_d1) / (2 * t_sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) *cdf_d1 - r * x * math.exp(-r * t) * cdf_d2
        vega = math.exp((b - r) * t) * fs * t_sqrt * pdf_d1
        rho = x * t * math.exp(-r * t) * cdf_d2

    else:
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(-d1)
        cdf_d2 = norm.cdf(-d2)

        value = x * math.exp(-r * t) * cdf_d2 - (fs * math.exp((b - r) * t) * cdf_d1)
        delta = -math.exp((b - r) * t) * cdf_d1
        gamma = math.exp((b - r) * t) * pdf_d1 / (fs * v * t_sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * pdf_d1) / (2 * t_sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * cdf_d1 + r * x * math.exp(-r * t) * cdf_d2
        vega = math.exp((b - r) * t) * fs * t_sqrt * pdf_d1
        rho = -x * t * math.exp(-r * t) * cdf_d2
        return {'value': value, 
                'delta': delta, 
                'gamma': gamma, 
                'theta': theta,  # theta here is the sensitivity relative to one year (365 days) 
                'vega': vega / 100,  # vega here is the sensitivity relative to 1% change in volatility
                'rho': rho / 100    # rho here is the sensitivity relative to 1% change in interest rate
                }

    
def black_scholes(option_type="call", *, fs, x, t, r, v):
    """ Black scholes model for stock options with no dividend
    :param option_type: one of ('call', 'put')
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params v: implied volatility (in percentage)
    """
    return _generalized_black_scholes(option_type=option_type, fs=fs, x=x, t=t, r=r, b=r, v=v)


def black_scholes_merton(option_type="call", *, fs, x, t, r, q, v):
    """ Black scholes merton model for stock options with dividend yield
    :param option_type: one of ('call', 'put')
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params q: dividend yield (in percentage)
    :params v: implied volatility (in percentage)
    """
    return  _generalized_black_scholes(option_type=option_type, fs=fs, x=x, t=t, r=r, b=r-q, v=v)


def black_scholes_commodity(option_type="call", *, fs, x, t, r, v):
    """ Black scholes model for commodities
    :param option_type: one of ('call', 'put')
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params v: implied volatility (in percentage)
    """
    return _generalized_black_scholes(option_type=option_type, fs=fs, x=x, t=t, r=r, b=0, v=v)


def garman_kohlhagen(option_type="call", *, fs, x, t, r, rf, v):
    """Garman-Kohlhagen model for FX options
    :param option_type: one of ('call', 'put')
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params rf: foreign interest rate (in percentage)
    :params v: implied volatility (in percentage)
    """
    return _generalized_black_scholes(option_type=option_type, fs=fs, x=x, t=t, r=r, b=r-rf, v=v)


def _implied_volatility(option_type="call", max_iter=500, min_volatility=0.01, max_volatility=1.0, xtol=0.001, *, p, fs, x, t, r, b):
    """
    :param option_type: one of ('call', 'put')
    :params p: price of the option
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params b: cost of carry (in percentage)
    """
    def obj(v):
        p_hat = _generalized_black_scholes(option_type=option_type, fs=fs, x=x, t=t, r=r, b=b, v=v)['value']
        return abs(p_hat - p)
    return scipy.optimize.bisect(obj, min_volatility, max_volatility, xtol=xtol, maxiter=max_iter)


def implied_volatility_stock(option_type="call", max_iter=500, min_volatility=0.01, max_volatility=1.0, xtol=0.001, *, p, fs, x, t, r, q):
    """ Find the implied volatility of stock option
    :param option_type: one of ('call', 'put')
    :params p: price of the option
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    :params q: dividend yield (in percentage)
    """
    b = r - q
    return _implied_volatility(option_type=option_type, max_iter=500, min_volatility=0.01, max_volatility=1.0, xtol=0.001, p=p, fs=fs, x=x, t=t, r=r, b=b)


def implied_volatility_commodity(option_type="call", max_iter=500, min_volatility=0.01, max_volatility=1.0, xtol=0.001, *, p, fs, x, t, r):
    """ Find the implied volatility of stock option
    :param option_type: one of ('call', 'put')
    :params p: price of the option
    :params fs: price of the underlying asset
    :params x: exercise price
    :params t: time to expiration in years 
    :params r: risk-free rate (in percentage)
    """
    return _implied_volatility(option_type=option_type, max_iter=500, min_volatility=0.01, max_volatility=1.0, xtol=0.001, p=p, fs=fs, x=x, t=t, r=r, b=0)
