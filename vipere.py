#! /usr/bin/env python3
## Licensed under a GPLv3 style license - see LICENSE
## vipere - Velocity and IP Estimator
## Copyright (C) Mathias Zechmeister and Jana Koehler

import argparse
import configparser
import glob
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import erf
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

c = 299792.458   # [km/s] speed of light
viperdir = os.path.dirname(os.path.realpath(__file__)) + os.sep
crires_path = viperdir + "lib/CRIRES/"


###############################################################################
# Gnuplot interface
###############################################################################

class Gplot(object):
   """
   An interface between Python and gnuplot.

   Creation of an instance opens a pipe to gnuplot and returns an object for communication.
   Plot commands are send to gnuplot via the call method; arrays as arguments are handled.
   Gnuplot options are set by calling them as method attributes.
   Each method returns the object again. This allows to chain set and plot method.

   Parameters
   ----------
   tmp : str, optional
       Method for passing data.
       * '$' - use inline datablock (default) (not faster than temporary data)
       * None - create a non-persistent temporary file
       * '' - create a local persistent file
       * '-' - use gnuplot special filename (no interactive zoom available,
               replot does not work)
       * 'filename' - create manually a temporary file
   stdout : boolean, optional
       If true, plot commands are send to stdout instead to gnuplot pipe.
   stderr : int, optional
       Gnuplot prints errors and user prints to stderr (term output is sent to stdout). The
       default stderr=None retains this behaviour. stderr=-1 (subprocess.PIPE) tries to
       capture stderr so that the output can be redirected to the Jupyter cells instead of
       parent console. This feature can be fragile and is experimental.
   mode : str, optional
       Primary command for the call method. The default is 'plot'. After creation it can
       be changed, e.g. gplot.mode = gplot.splot.

   args : array or str for function, file, or other plot commands like style
   flush : str, optional
       set to '' to suppress flush until next the ogplot (for large data sets)

   Examples
   --------

   A simple plot and add a data set

   >>> gplot('sin(x) w lp lt 2')
   >>> gplot+(np.arange(10)**2., 'w lp lt 3')
   >>> gplot+('"filename" w lp lt 3')

   Pass multiple curves in one call

   >>> gplot('x/5, 1, x**2/50 w l lt 3,', np.sqrt(np.arange(10)),' us 0:1 ps 2 pt 7 ,sin(x)')
   >>> gplot.mxtics().mytics(2).repl
   >>> gplot([1,2,3,4])
   >>> gplot([1,2,3,4], [2,3,1,1.5])
   >>> gplot([1,2,3,4], [[2,2,1,1.5], [3,1,4,5.5]])
   >>> gplot([[2,2,1,1.5]])
   >>> gplot([1],[2],[3],[4])
   >>> gplot(1,2,3,4)

   Pass options as function arguments or in the method name separated with underscore

   >>> gplot.key_bottom_rev("left")('sin(x)')
   """
   version = subprocess.check_output(['gnuplot', '-V'])
   version = float(version.split()[1])

   def __init__(self, cmdargs='', tmp='$', mode='plot', stdout=False, stderr=None):
      self.stdout = stdout
      self.tmp = tmp
      self.mode = getattr(self, mode)   # set the default mode for __call__ (plot, splot)
      self.gnuplot = subprocess.Popen('gnuplot '+cmdargs, shell=True, stdin=subprocess.PIPE,
                   stderr=stderr, universal_newlines=True, bufsize=0)
      self.pid = self.gnuplot.pid
      self.og = 0   # overplot number
      self.buf = ''
      self.tmp2 = []
      self.flush = None
      self.put = self._put
      if stderr:
          import fcntl
          fcntl.fcntl(self.gnuplot.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
          self.put = self.PUT

   def _plot(self, *args, **kwargs):
      # collect all arguments
      tmp = kwargs.pop('tmp', self.tmp)
      flush = kwargs.pop('flush', '\n')
      if self.version in [4.6] and flush=="\n": flush = "\n\n"
      self.flush = flush
      pl = ''
      buf = ''
      data = ()
      if tmp in ('$',):
           pl, self.buf = self.buf, pl

      for arg in args + (flush,):
         if isinstance(arg, (str, u''.__class__)):
            if data:
               data = zip(*data)
               self.og += 1
               tmpname = tmp
               if tmp in ('-',):
                  self.buf += "\n".join(" ".join(map(str,tup)) for tup in data)+"\ne\n"
               elif tmp in ('$',):
                  tmpname = "$data%s" % self.og
                  buf += tmpname+" <<EOD\n"+("\n".join(" ".join(map(str,tup)) for tup in data))+"\nEOD\n"
               elif tmp is None:
                  self.tmp2.append(tempfile.NamedTemporaryFile())
                  tmpname = self.tmp2[-1].name
                  np.savetxt(self.tmp2[-1], list(data), fmt="%s")
                  self.tmp2[-1].seek(0)
               else:
                  if tmp == '':
                     tmpname = 'gptmp_'+str(self.pid)+str(self.og)
                  np.savetxt(tmpname, list(data), fmt="%s")
               pl += '"'+tmpname+'"'
            pl += arg
            data = ()
         else:
            _1D = hasattr(arg, '__iter__')
            _2D = _1D and hasattr(arg[0], '__iter__') and not isinstance(arg[0], str)
            data += tuple(arg) if _2D else (arg,) if _1D else ([arg],)

      if tmp in ('$',):
          self.put(buf, end='')
          self.buf += pl
          pl = ''
      self.put(pl, end='')
      if flush != '':
          self.put(self.buf, end='')
          self.buf = ''

   def _put(self, *args, **kwargs):
      print(file=None if self.stdout else self.gnuplot.stdin, *args, **kwargs)
      return self

   def PUT(self, *args, **kwargs):
      self._put(*args, **kwargs)
      self.gnuplot.stdin.write('printerr ""\n')
      import time as _time
      try:
          _time.sleep(0.03)
          s = self.gnuplot.stderr.read()[:-1]
          if s: print(s, end='')
      except:
          pass
      return self

   def plot(self, *args, **kwargs):
      self.og = 0; self.buf = ''; self.put('\n')
      return self._plot('plot ', *args, **kwargs)

   def splot(self, *args, **kwargs):
      self.og = 0; self.buf = ''; self.put('\n')
      return self._plot('splot ', *args, **kwargs)

   def replot(self, *args, **kwargs):
      return self._plot('replot ', *args, **kwargs)

   def test(self, *args, **kwargs):
      return self._plot('test', *args, **kwargs)

   def oplot(self, *args, **kwargs):
      pl = ',' if self.flush=='' else ' replot '
      return self._plot(pl, *args, **kwargs)

   def array(self, **kwargs):
      for k,v in kwargs.items(): self.put("array %s[%d] = %s" % (k, len(v), list(v)))
      return self

   def var(self, **kwargs):
      for i in kwargs.items(): self.put("%s=%s" % i)
      return self

   def __call__(self, *args, **kwargs):
      return self.mode(*args, **kwargs)

   def __getattr__(self, name):
      if name in ('__repr__', '__str__'):
         raise AttributeError
      elif name=='repl':
         return self.replot()
      elif name.startswith(('load', 'pwd', 'set', 'show', 'system', 'unset', 'reset', 'print', 'bind')):
         def func(*args):
            return self.put(name.replace("_"," "), *args)
         return func
      else:
         def func(*args):
            return self.set(name.replace("_"," "), *args)
         return func

   def __add__(self, other):
      self.oplot(*other) if other else self._plot()

   def __sub__(self, other):
      self(*other, flush='')

   def __lt__(self, other):
      self.oplot(*other, flush='')


gplot = Gplot()
ogplot = gplot.oplot
gplot.colors('classic')
gplot2 = Gplot()


###############################################################################
# Parameter handling
###############################################################################

class param(float):
    '''
    Parameter with uncertainty property.

    Examples
    --------
    >>> p = param(5, 0)
    >>> p
    5 ± 0
    >>> f'{p:06.3f}'
    '05.000'
    >>> p + 2
    7.0
    '''
    def __new__(cls, value, unc=None):
        instance = super().__new__(cls, value)
        instance.unc = unc
        instance.value = value
        return instance

    def __repr__(self):
        return f'{self.value}' + ('' if self.unc is None  else f' ± {self.unc}')


class nesteddict(dict):
    '''
    A named and nested dictionary with multi-dimensional indexing.
    '''
    __getattr__ = dict.__getitem__
    values = lambda _: [*super().values()]
    keys = lambda _: [*super().keys()]

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self[key[0]][key[1]]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self[key[0]][key[1]] = value
        else:
            super().__setitem__(key, value)

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, *args, **kwargs):
        for d in args + (kwargs,):
            for k in d:
                self[k] = d[k]

    def flat(self):
        d = {}
        for key, values in self.items():
            if isinstance(values, list):
                for (i, val) in enumerate(values):
                    d[(key, i)] = val
            elif isinstance(values, dict):
                for (k, val) in values.items():
                    d[(key, k)] = val
            else:
               d[key] = values
        return d

    def __add__(self, d):
        p = self.__class__(self, d)
        return p

    def __repr__(self):
        return "\n".join([f'{k}: '+repr(v).replace('\n',', ') for k,v in self.items()])


class Params(nesteddict):
    '''
    A collection/group of param.
    '''
    def __setitem__(self, key, value):
        super().__setitem__(key, self._as_param(value))

    def _as_param(self, value):
        if isinstance(value, (param, Params)):
             p = value
        elif isinstance(value, (float, int)):
             p = param(value)
        elif isinstance(value, tuple):
             p = param(*value)
        elif type(value).__name__ in ('list', 'ndarray'):
             p = [self._as_param(val) for val in value]
        elif isinstance(value, dict):
             p = Params(value)
        else:
            print(value, type(value), 'not supported')
        return p

    def vary(self):
        return {k: v for k,v in self.flat().items() if v.unc != 0}


###############################################################################
# FTS resampling
###############################################################################

def FTSfits(ftsname):

    if ftsname.endswith(".dat"):
        data = np.loadtxt(ftsname)
        w = data[:, 0]
        f = data[:, 1]
        f = f[::-1]
        w = 1e8 / w[::-1]
    elif ftsname.endswith(".fits"):
        hdu = fits.open(ftsname, ignore_blank=True, output_verify='silentfix')

        hdr = hdu[0].header
        cdelt1 = hdr.get('CDELT1', 'none')

        if cdelt1 == 'none':
            wavetype = hdr.get('wavetype', 'none')
            unit = hdr.get('unit', 'none')
            w = hdu[1].data['wave']
            f = hdu[1].data['flux']

            if wavetype == 'wavenumber':  w = 1e8 / w[::-1]
            if unit == 'nm': w *= 10

        else:
            f = hdu[0].data[::-1]
            try:
                w = hdr['CRVAL1'] + hdr['CDELT1'] * (np.arange(f.size) + 1. - hdr['CRPIX1'])
            except:
                w = hdr['CRVAL1'] + hdr['CDELT1'] * (np.arange(f.size) + 1.)
            w = 1e8 / w[::-1]

    return w, f


def resample(w, f, dv=100):
    '''
    dv: Sampling step for uniform log(lambda) [m/s]
    '''
    lnw = np.log(w)
    lnwj = np.arange(lnw[0], lnw[-1], dv / (c * 1000))
    iod_j = np.interp(lnwj, lnw, f)
    return w, f, lnwj, iod_j


###############################################################################
# IP functions and forward model
###############################################################################

def IP(vk, s=2.2):
    """Gaussian IP"""
    IP_k = np.exp(-(vk/s)**2/2)
    IP_k /= IP_k.sum()
    return IP_k

def IP_sg(vk, s=2.2, e=2.):
    """super Gaussian"""
    IP_k = np.exp(-abs(vk/s)**e)
    IP_k /= IP_k.sum()
    return IP_k

def IP_ag(vk, s=2.2, a=0):
    '''Asymmetric (skewed) Gaussian.'''
    b = a / np.sqrt(1+a**2) * np.sqrt(2/np.pi)
    ss = s / np.sqrt(1-b**2)
    vk = (vk + ss*b) / ss
    IP_k = np.exp(-vk**2/2) * (1+erf(a/np.sqrt(2)*vk))
    IP_k /= IP_k.sum()
    return IP_k

def IP_agr(vk, s, a=0):
    a = 10 * np.tanh(a/10)
    return IP_ag(vk, s, a=a)

def IP_asg(vk, s=2.2, e=2., a=1):
    """asymmetric super Gaussian"""
    mu = 0
    for _ in range(2):
        IP_k = np.exp(-abs((vk+mu)/s)**e)
        IP_k *= (1+erf(a/np.sqrt(2) * (vk+mu)))
        IP_k /= IP_k.sum()
        mu += IP_k.dot(vk)
    return IP_k

def IP_sbg(vk, s1=2.2, s2=1, e=2.):
    """super bi-Gaussian"""
    IP_k = np.exp(-abs((vk+mu)/s)**e)
    IP_k *= (1+erf(a/np.sqrt(2) * (vk+mu)))
    IP_k /= IP_k.sum()
    return IP_k

def IP_bg(vk, s1=2., s2=2.):
    """BiGaussian"""
    xc = np.sqrt(2/np.pi) * (-s1**2 + s2**2) / (s1+s2)
    vck = vk + xc
    IP_k = np.exp(-0.5*(vck/np.where(vck<0, s1, s2))**2)
    IP_k /= IP_k.sum()
    return IP_k

def IP_mcg(vk, s0=2, a1=0.1):
    """IP for multiple, central Gaussians."""
    s1 = 4 * s0
    a1 = a1 / 10
    IP_k = np.exp(-(vk/s0)**2)
    IP_k += a1 * np.exp(-(vk/s1)**2)
    IP_k = IP_k.clip(0, None)
    IP_k /= IP_k.sum()
    return IP_k

def IP_mg(vk, *a):
    """IP for multiple uniformly spaced Gaussians ("Gaussian spline")."""
    s = 0.9
    dx = s
    na = len(a) + 1
    mid = len(a) // 2
    a = np.tanh(a)
    a = [*a[:mid], 1, *a[mid:]]
    xl = np.arange(na)
    xm = np.dot(xl, a) / sum(a)
    xc = (dx * (xl-xm))[:, np.newaxis]
    IP_k = np.exp(-((vk-xc)/s)**2)
    IP_k = np.dot(a, IP_k)
    IP_k /= IP_k.sum()
    return IP_k

def IP_lor(vk, s=2.2):
    """Lorentzian IP"""
    IP_k = 1 / np.pi* np.abs(s) / (s**2+vk**2)
    IP_k /= IP_k.sum()
    return IP_k

IPs = {'g': IP, 'sg': IP_sg, 'sbg': IP_sbg, 'ag': IP_ag, 'agr': IP_agr, 'asg': IP_asg, 'bg': IP_bg, 'mg': IP_mg, 'mcg': IP_mcg, 'lor': IP_lor, 'bnd': 'bnd'}


def poly(x, a):
    return np.polyval(a[::-1], x)

def pade(x, a, b):
    '''
    rational polynomial
    b: denominator coefficients b1, b2, ... (b0 is fixed to 1)
    '''
    y = poly(x, a) / (1+x*poly(x, b))
    return y


class model:
    '''
    The forward model.
    '''
    def __init__(self, *args, func_norm=poly, IP_hs=50, xcen=0):
        self.xcen = xcen
        self.S_star, self.lnwave_j, self.spec_cell_j, self.fluxes_molec, self.IP = args
        self.dx = self.lnwave_j[1] - self.lnwave_j[0]
        self.IP_hs = IP_hs
        self.vk = np.arange(-IP_hs, IP_hs+1) * self.dx * c
        self.lnwave_j_eff = self.lnwave_j[IP_hs:-IP_hs]
        self.func_norm = func_norm

    def __call__(self, pixel, rv=0, norm=[1], wave=[], ip=[], atm=[], bkg=[0], ipB=[]):
        coeff_norm, coeff_wave, coeff_ip, coeff_atm, coeff_bkg, coeff_ipB = norm, wave, ip, atm, bkg, ipB

        spec_gas = 1 * self.spec_cell_j

        if len(self.fluxes_molec):
            flux_atm = np.nanprod(np.power(self.fluxes_molec, np.abs(coeff_atm[:len(self.fluxes_molec)])[:, np.newaxis]), axis=0)
            if len(coeff_atm) == len(self.fluxes_molec)+1:
                flux_atm = np.interp(self.lnwave_j, self.lnwave_j-np.log(1+coeff_atm[-1]/c), flux_atm)
            spec_gas *= flux_atm

        Sj_eff = np.convolve(self.IP(self.vk, *coeff_ip), self.S_star(self.lnwave_j-rv/c) * (spec_gas + coeff_bkg[0]), mode='valid')

        if len(coeff_ipB):
            coeff_ipB = [coeff_ipB[0]*coeff_ip[0], *coeff_ip[1:]]
            Sj_B = np.convolve(self.IP(self.vk, *coeff_ipB), self.S_star(self.lnwave_j-rv/c) * (spec_gas + coeff_bkg[0]), mode='valid')
            Sj_A = Sj_eff
            g = self.lnwave_j_eff - self.lnwave_j_eff[0]
            g /= g[-1]
            Sj_eff = (1-g)*Sj_A + g*Sj_B

        lnwave_obs = np.log(poly(pixel-self.xcen, coeff_wave))
        Si_eff = np.interp(lnwave_obs, self.lnwave_j_eff, Sj_eff)
        Si_mod = self.func_norm(pixel-self.xcen, coeff_norm) * Si_eff
        return Si_mod

    def fit(self, pixel, spec_obs, par, sig=[], **kwargs):
        '''Generic fit wrapper.'''
        varykeys, varyvals = zip(*par.vary().items())
        S_model = lambda x, *params: self(x, **(par + dict(zip(varykeys, params))))
        params, e_params = curve_fit(S_model, pixel, spec_obs, p0=varyvals, sigma=sig, absolute_sigma=False, epsfcn=1e-12)
        pnew = par + dict(zip(varykeys, params))
        for k, v in zip(varykeys, np.sqrt(np.diag(e_params))):
            pnew[k].unc = v
        if kwargs:
            self.show(pnew, pixel, spec_obs, par_rv=pnew.rv, **kwargs)
        return pnew, e_params

    def show(self, params, x, y, par_rv=None, res=True, x2=None, dx=None, rel_fac=None):
        '''
        res: Show residuals.
        x2: Values for second x axis.
        rel_fac: Factor for relative residuals.
        dx: Subpixel step size for the model [pixel].
        '''
        ymod = self(x, **params)
        if x2 is None:
            x2 = np.poly1d(params.wave[::-1])(x-self.xcen)
        if par_rv:
            gplot.RV2title(", v=%.2f ± %.2f m/s" % (par_rv*1000, par_rv.unc*1000))
        gplot.put("if (!exists('lam')) {lam=1}")
        gplot.key('horizontal')
        gplot.xlabel('lam?"Vacuum wavelength [Å]":"Pixel x"')
        gplot.ylabel('"flux"')
        gplot.bind('"$" "lam=!lam; set xlabel lam?\\"Vacuum wavelength [Å]\\":\\"Pixel x\\"; replot"')
        args = (x, y, ymod, x2, 'us lam?4:1:2:3 w lp pt 7 ps 0.5 t "obs",',
          '"" us lam?4:1:3 w p pt 6 ps 0.5 lc 3 t "model"')
        prms = np.nan
        if dx:
            xx = np.arange(x.min(), x.max(), dx)
            xx2 = np.poly1d(params.wave[::-1])(xx-self.xcen)
            yymod = self(xx, **params)
            args += (",", xx, yymod, xx2, 'us lam?3:1:2 w l lc 3 t ""')
        if res or rel_fac:
            col2 = rel_fac * np.mean(ymod) * (y/ymod - 1) if rel_fac else y - ymod
            rms = np.std(col2)
            prms = rms / np.mean(ymod) * 100
            gplot.mxtics().mytics().my2tics()
        if res:
            args += (",", x, col2, x2, "us lam?3:1:2 w p pt 7 ps 0.5 lc 1 t 'res (%.3g \~ %.3g%%)', 0 lc 3 t ''" % (rms, prms))
        if rel_fac:
            args += (",", x, col2, x2, "us lam?3:1:2 w l lc 1 t 'res (%.3g \~ %.3g%%)', 0 lc 3 t ''" % (rms, prms))
        if res or rel_fac or dx:
            gplot.yrange("[:%g]" % (1.4*np.nanmax(ymod)))
            gplot(*args)
        return prms


class model_bnd(model):
    '''
    The forward model with band matrix.
    '''
    def __init__(self, *args, func_norm=poly, IP_hs=50, xcen=0):
        self.xcen = xcen
        self.S_star, self.lnwave_j, self.spec_cell_j, self.IP = args
        self.dx = self.lnwave_j[1] - self.lnwave_j[0]
        self.IP_hs = IP_hs
        self.vk = np.arange(-IP_hs, IP_hs+1) * self.dx * c
        self.lnwave_j_eff = self.lnwave_j[IP_hs:-IP_hs]
        self.func_norm = func_norm

    def base(self, x=None, degk=3, sig_k=1):
        '''Setup the base functions.'''
        self.x = x
        bg = self.IP
        lnwave_obs = np.log(np.poly1d(bg[::-1])(x-self.xcen))
        j = np.arange(self.lnwave_j.size)
        jx = np.interp(lnwave_obs, self.lnwave_j, j)
        vl = np.array([-1.4, -0.7, 0, 0.7, 1.4])[np.newaxis,np.newaxis,:]
        self.bnd = jx[:,np.newaxis].astype(int) + np.arange(-self.IP_hs, self.IP_hs+1)
        self.BBxjl = np.exp(-(self.lnwave_j[self.bnd][...,np.newaxis]-lnwave_obs[:,np.newaxis,np.newaxis]+sig_k*vl)**2/sig_k**2)
        self.Bxk = np.vander(x-x.mean(), degk)[:,::-1]

    def Axk(self, v, **kwargs):
        if kwargs:
            self.base(**kwargs)
        starj = self.S_star(self.lnwave_j-v/c)
        _Axkl = np.einsum('xj,xjl,xk->xkl', (starj*self.spec_cell_j)[self.bnd], self.BBxjl, self.Bxk)
        return _Axkl

    def IPxj(self, akl, **kwargs):
        if kwargs:
            self.base(**kwargs)
        IPxj = np.einsum('xjl,xk,kl->xj', self.BBxjl, self.Bxk, akl.reshape(self.Bxk.shape[1], -1))
        return IPxj

    def fit(self, f, v, **kwargs):
        Axkl = self.Axk(v, **kwargs)
        return np.linalg.lstsq(Axkl.reshape((len(Axkl), -1)), f, rcond=1e-32)

    def __call__(self, x, v, ak, **kwargs):
        Axkl = self.Axk(v, **kwargs)
        fx = Axkl.reshape((len(Axkl), -1)) @ ak
        return fx


def show_model(x, y, ymod, res=True):
    gplot(x, y, ymod, 'w lp pt 7 ps 0.5 t "S_i",',
          '"" us 1:3 w lp pt 6 ps 0.5 lc 3 t "S(i)"')
    if res:
        rms = np.std(y-ymod)
        gplot.mxtics().mytics().my2tics()
        gplot.y2range('[-0.2:2]').ytics('nomirr').y2tics()
        gplot+(x, y-ymod, "w p pt 7 ps 0.5 lc 1 axis x1y2 t 'res %.3g', 0 lc 3 axis x1y2" % rms)


###############################################################################
# CRIRES instrument
###############################################################################

crires_location = EarthLocation.from_geodetic(
    lat=-24.6268 * u.deg, lon=-70.4045 * u.deg, height=2648 * u.m
)

oset = '1:28'  # covers up to 9 orders/det (Y/J band); K/H use fewer via -oset
ip_guess = {'s': 1.5}
fts_default = 'lib/CRIRES/FTS/CRp_SGC2_FTStmpl-HR0p007-WN3000-5000_Kband.dat'


def Spectrum(filename='', order=None, targ=None):

    order_idx, detector = divmod(order-1, 3)
    detector += 1

    exptime = 0

    hdu = fits.open(filename, ignore_blank=True)
    hdr = hdu[0].header
    ra = hdr.get('RA', np.nan)
    de = hdr.get('DEC', np.nan)
    setting = hdr['ESO INS WLEN ID']
    nod_type = hdr['ESO PRO CATG']
    cal = hdr['ESO PRO REC1 CAL* CATG']

    try:
        if str(nod_type) != 'OBS_NODDING_EXTRACT_COMB': raise
        dateobs = Time(hdr["ESO DRS TMID"], format='mjd').isot
    except:
        dateobs = hdr['DATE-OBS']
        ndit = hdr.get('ESO DET NDIT', 1)
        nods = hdr.get('ESO PRO DATANCOM', 1)
        if str(nod_type) in ('OBS_NODDING_EXTRACTA', 'OBS_NODDING_EXTRACTB'):
            nods /= 2
        exptime = hdr.get('ESO DET SEQ1 DIT', 0)
        exptime = (exptime*nods*ndit) / 2.0

    n_orders = max(int(cc) for cc in [col.split('_')[0] for col in hdu[detector].columns.names if col.endswith('_SPEC')])
    order_drs = n_orders - order_idx
    err = hdu[detector].data["0"+str(order_drs)+"_01_ERR"]
    spec = hdu[detector].data["0"+str(order_drs)+"_01_SPEC"]

    pixel = np.arange(spec.size)

    targdrs = SkyCoord(ra=ra*u.deg, dec=de*u.deg)
    if not targ: targ = targdrs
    midtime = Time(dateobs, format='isot', scale='utc') + exptime * u.s
    berv = targ.radial_velocity_correction(obstime=midtime, location=crires_location)
    berv = berv.to(u.km/u.s).value
    bjd = midtime.tdb

    if 0: #str(setting) in ('K2148', 'K2166', 'K2192'):
        file_wls = np.genfromtxt(crires_path+'wavesolution_own/wave_solution_'+str(setting)+'.dat', dtype=None, names=True).view(np.recarray)
        coeff_wls = [file_wls.b1[order-1], file_wls.b2[order-1], file_wls.b3[order-1]]
        wave = np.poly1d(coeff_wls[::-1])(pixel)
    else:
        wave = (hdu[detector].data["0"+str(order_drs)+"_01_WL"]) * 10

    if 'CAL_FLAT_EXTRACT_1D' not in str(cal):
        try:
            hdu_blaze = fits.open(crires_path+'blaze_own.fits', ignore_blank=True)
            blaze = hdu_blaze[setting].data["0"+str(order_drs)+"_0"+str(detector)+"_BLAZE"]
            spec /= blaze
        except (KeyError, IndexError):
            pass

    flag_pixel = 1 * np.isnan(spec)

    return pixel, wave, spec, err, flag_pixel, bjd, berv


def Tpl(tplname, order=None, targ=None):
    '''Tpl should return barycentric corrected wavelengths'''

    if tplname.endswith('_tpl.fits'):
        order_idx, detector = divmod(order-1, 3)
        detector += 1

        hdu = fits.open(tplname, ignore_blank=True)
        hdr = hdu[0].header
        n_orders = max(int(cc) for cc in [col.split('_')[0] for col in hdu[detector].columns.names if col.endswith('_SPEC')])
        order_drs = n_orders - order_idx
        err = hdu[detector].data["0"+str(order_drs)+"_01_ERR"]
        spec = hdu[detector].data["0"+str(order_drs)+"_01_SPEC"]
        wave = hdu[detector].data["0"+str(order_drs)+"_01_WL"]
    else:
        pixel, wave, spec, err, flag_pixel, bjd, berv = Spectrum(tplname, order=order, targ=targ)
        wave *= 1 + berv/c

    return wave, spec


def FTS(ftsname=fts_default, dv=100):
    return resample(*FTSfits(ftsname), dv=dv)


def write_fits(wtpl_all, tpl_all, e_all, list_files, file_out):

    file_in = list_files[0]

    hdu = fits.open(file_in, ignore_blank=True)
    hdr = hdu[0].header

    if len(list_files) > 1:
        del hdr['DATE-OBS']
        del hdr['UTC']
        del hdr['LST']
        del hdr['ARCFILE']
        del hdr['ESO INS SENS*']
        del hdr['ESO INS TEMP*']
        del hdr['ESO INS1*']
        del hdr['ESO DET*']
        del hdr['ESO OBS*']
        del hdr['ESO TPL*']
        del hdr['ESO TEL*']
        del hdr['ESO OCS MTRLGY*']
        del hdr['ESO ADA*']
        del hdr['ESO AOS*']
        del hdr['ESO SEQ*']
        del hdr['ESO PRO DATANCOM']
        del hdr['ESO PRO REC1 PARAM*']
        del hdr['ESO PRO REC1 RAW*']

        for hdri in hdu:
            hdri.header['EXPTIME'] = 0

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%dT%H:%M:%S")
    hdr['DATE'] = dt_string

    hdr.set('HIERARCH ESO PRO REC2 ID', 'viper_create_tpl', 'Pipeline recipe', after='ESO PRO REC1 PIPE ID')

    for i in range(0, len(list_files), 1):
        pathi, filei = os.path.split(list_files[len(list_files)-i-1])
        hdr.set('HIERARCH ESO PRO REC2 RAW'+str(len(list_files)-i)+' NAME', filei, 'File name', after='ESO PRO REC2 ID')

    hdr.set('HIERARCH ESO PRO DATANCOM2', len(list_files), 'Number of combined frames', after='ESO PRO REC2 RAW'+str(len(list_files))+' NAME')

    for detector in (1, 2, 3):
        data = hdu[detector].data
        cols = hdu[detector].columns
        n_orders = max(int(cc.name.split('_')[0]) for cc in cols if cc.name.endswith('_SPEC'))

        for cc in cols[::3]:
            odrs = int(cc.name.split('_')[0])
            o = (n_orders-odrs)*3 + detector
            if o in list(tpl_all.keys()):
                data["0"+str(odrs)+"_01_WL"] = wtpl_all[o]
                data["0"+str(odrs)+"_01_SPEC"] = tpl_all[o]
                data["0"+str(odrs)+"_01_ERR"] = e_all[o]
            else:
                wave0 = data["0"+str(odrs)+"_01_WL"]
                npix = len(wave0)
                data["0"+str(odrs)+"_01_WL"] = wave0 * 10
                data["0"+str(odrs)+"_01_SPEC"] = np.ones(npix)
                data["0"+str(odrs)+"_01_ERR"] = np.nan * np.ones(npix)

    hdu.writeto(file_out+'_tpl.fits', overwrite=True)
    hdu.close()


###############################################################################
# Main pipeline
###############################################################################

modset = {}
insts = ['CRIRES']

class nameddict(dict):
    __getattr__ = dict.__getitem__
    def translate(self, x):
        return [name for name,f in self.items() if (f & x) or f==x==0]

flag = nameddict(
    ok=       0,
    nan=      1,
    neg=      2,
    sat=      4,
    atm=      8,
    sky=     16,
    out=     32,
    clip=    64,
    lowQ=   128,
    badT=   256,
    chunk=  512,
)


def arg2slice(arg):
    """Convert string argument to a slice."""
    if isinstance(arg, str):
        arg = eval('np.s_['+arg+']')
    return [arg] if isinstance(arg, int) else arg

def arg2range(arg):
    return  eval('np.r_['+arg+']')


def SSRstat(vgrid, SSR, dk=1, plot='maybe', N=None):
    '''
    Analyse chi2 peak.

    Parameters
    ----------
    N: Number of data points in the fit. Needed to estimate 1 sigma uncertainty and when SSR are not chi2 values.
    '''
    k = np.argmin(SSR[dk:-dk]) + dk
    vpeak = vgrid[k-dk:k+dk+1]
    SSRpeak = SSR[k-dk:k+dk+1] - SSR[k]
    v_step = vgrid[1] - vgrid[0]
    a = np.array([0, (SSR[k+dk]-SSR[k-dk])/(2*v_step), (SSR[k+dk]-2*SSR[k]+SSR[k-dk])/(2*v_step**2)])
    v = (SSR[k+dk]-SSR[k-dk]) / (SSR[k+dk]-2*SSR[k]+SSR[k-dk]) * 0.5 * v_step
    v = vgrid[k] - a[1]/2./a[2]
    e_v = np.nan
    if -1 in SSR:
        print('opti warning: bad ccf.')
    elif a[2] <= 0:
        print('opti warning: a[2]=%f<=0.' % a[2])
    elif not vgrid[0] <= v <= vgrid[-1]:
        print('opti warning: v not in [va,vb].')
    else:
        e_v = 1. / a[2]**0.5
        if N:
            SSRmin = SSR[k] + a[0] - a[1] * a[1]/2./a[2] + a[2] * (a[1]/2./a[2]) **2
            e_v *= (SSRmin/N) **0.5

    if (plot==1 and np.isnan(e_v)) or plot==2:
        gplot2.yrange('[*:%f]' % np.max(SSR))
        gplot2(vgrid, SSR-SSR[k], " w lp, vk="+str(vgrid[k])+", %f+(x-vk)*%f+(x-vk)**2*%f," % tuple(a), [v,v], [0,SSR[1]], 'w l t "%f km/s"'%v)
        gplot2+(vpeak, SSRpeak, ' lt 1 pt 6; set yrange [*:*]')
    return v, e_v, a


if __name__ == "__main__" or __name__ == "vipere":
    argparse.ArgumentDefaultsHelpFormatter._split_lines = lambda self, text, width: text.splitlines()

    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument('args', nargs='*')
    preparser.add_argument('-inst', help='Instrument.', default='CRIRES', choices=insts)
    preparser.add_argument('-config_file', help='Config file and optional section  [None DEFAULT].', nargs='*', type=str)
    preargs = preparser.parse_known_args()[0]

    Tell = None
    iset = slice(None)

    # read in default values from config_viper.ini
    configs_inst, configs_user = {}, {}
    config_default = configparser.ConfigParser()
    config_default.read(viperdir+'config_viper.ini')
    configs_def = dict(config_default['DEFAULT'])
    if preargs.inst in config_default.sections():
        configs_inst = dict(config_default[preargs.inst])

    if preargs.config_file:
        if len(preargs.config_file) == 1 and not preargs.config_file[0].endswith('.ini'):
            if preargs.config_file[0] in config_default.sections():
                configs_user = dict(config_default[preargs.config_file[0]])
        elif len(preargs.config_file) == 2:
            config = configparser.ConfigParser()
            config.read(preargs.config_file[0])
            if preargs.config_file[1] in config.sections():
                configs_user = dict(config[preargs.config_file[1]])
            else:
                print('WARNING: Declared section is not found in %s. Use DEFAULT values instead.' % preargs.config_file[0])

    parser = argparse.ArgumentParser(description='VIPER - velocity and IP Estimator', add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argopt = parser.add_argument
    argopt('obspath', help='Filename of observation.', default='data/TLS/betgem/BETA_GEM.fits', type=str)
    argopt('tplname', help='Filename of template.', nargs='?', type=str)
    argopt('-inst', help='Instrument.', default='CRIRES', choices=insts)
    argopt('-fts', help='Filename of FTS Cell.', default=viperdir + fts_default, dest='ftsname', type=str)
    argopt('-ip', help='IP model (g: Gaussian, ag: asymmetric (skewed) Gaussian, sg: super Gaussian, bg: biGaussian, mg: multiple Gaussians, mcg: multiple central Gaussians, bnd: bandmatrix).', default='g', choices=[*IPs], type=str)
    argopt('-chunks', nargs='?', help='Divide one order into a number of chunks.', default=1, type=int)
    argopt('-config_file', nargs='*', help='Config file and optional section  [None DEFAULT].', type=str)
    argopt('-createtpl', nargs='?', help='Removal of telluric features (or cell lines) and combination of several observations.', default=False, const=True, type=int)
    argopt('-deg_bkg', nargs='?', help='Number of additional parameters.', default=0, const=1, type=int)
    argopt('-deg_norm', nargs='?', help='Polynomial degree for flux normalisation.', default=3, type=int)
    argopt('-deg_norm_rat', nargs='?', help='Rational polynomial degree of denominator for flux normalisation.', type=int)
    argopt('-deg_wave', nargs='?', help='Polynomial degree for wavelength scale l(x).', default=3, type=int)
    argopt('-fix', nargs='*', help='Fix parameter. -fix wave will fix wavelength, as needed for stabilized instruments (like TRES, HARPS).', default=['None'], type=str)
    argopt('-flagfile', help='Use just good region as defined in flag file.', default='', type=str)
    argopt('-iphs', nargs='?', help='Half size of the IP.', default=50, type=int)
    argopt('-ipB', nargs='*', help='Factor of IP width varation.', type=float, default=[])
    argopt('-iset', help='Pixel range.', default=iset, type=arg2slice)
    argopt('-kapsig', nargs='*', help='Kappa sigma values for the clipping stages. Zero does not clip.', default=[0], type=float)
    argopt('-kapsig_ctpl', help='Kappa sigma values for the clipping of outliers in template creation.', default=0.6, type=float)
    argopt('-look', nargs='?', help='See final fit of chunk with pause.', default=[], const=':200', type=arg2range)
    argopt('-lookfast', nargs='?', help='See final fit of chunk without pause.', default=[], const=':200', type=arg2range)
    argopt('-molec', nargs='*', help='Molecular specifies; all: Automatic selection of all present molecules.', default=['all'], type=str)
    argopt('-nexcl', nargs='*', help='Ignore spectra with string pattern.', default=[], type=str)
    argopt('-nocell', help='Do the calibration without using the FTS.', action='store_true')
    argopt('-nset', help='Index for spectrum.', default=':', type=arg2slice)
    argopt('-oset', help='Index for order.', default=oset, type=arg2slice)
    argopt('-oversampling', help='Oversampling factor for the template data.', default=None, type=int)
    argopt('-rv_guess', help='RV guess.', default=1., type=float)
    argopt('-tag', help='Output tag for filename.', default='tmp', type=str)
    argopt('-tellshift', nargs='?', help='Variable telluric wavelength shift (one value for all selected molecules).', default=False, const=True, type=int)
    argopt('-telluric', help='Treating tellurics (add: telluric forward modelling with one coeff for each molecule; add2: telluric forward modelling with combined coeff for non-water molecules).', default='', choices=['', 'add', 'add2'], type=str)
    argopt('-tpl_noRV', nargs='?', help='No stellar RV shift is applied to the telluric corrected spectrum. Just in combination with -createtpl.', default=False, const=True, type=int)
    argopt('-tpl_wave', help='Output wavelength of generated template (initial: take wavelengths from imput file; berv: apply barycentric correction to input wavelengths; tell: updated wavelength solution estimated via telluric lines).', default='initial', type=str)
    argopt('-tsig', help='(Relative) sigma value for weighting tellurics.', default=1, type=float)
    argopt('-vcut', help='Trim the observation to a range valid for the model [km/s]', default=100, type=float)
    argopt('-wgt', nargs='?', help='Weighted least square fit (error: employ data error; tell: upweight tellurics and downweight stellar lines)', default='', type=str)
    argopt('-?', '-h', '-help', '--help', help='Show this help message and exit.', action='help')

    parser.set_defaults(**configs_def)
    parser.set_defaults(**configs_inst)
    parser.set_defaults(**configs_user)

    parser.set_defaults(kapsig = [float(i) for i in (argopt('--kapsig').default.split(' '))])

    args = parser.parse_args()
    globals().update(vars(args))


def fit_chunk(order, chunk, obsname, tpltarg=None):
    ####  observation  ####
    pixel, wave_obs, spec_obs, err_obs, flag_obs, bjd, berv = Spectrum(obsname, order=order)

    flag_obs[np.isnan(spec_obs)] |= flag.nan

    lmin = max(wave_obs[iset][0], wave_tpl[order][0], wave_cell[0])
    lmax = min(wave_obs[iset][-1], wave_tpl[order][-1], wave_cell[-1])

    flag_obs[np.log(wave_obs) < np.log(lmin)+vcut/c] |= flag.out
    flag_obs[np.log(wave_obs) > np.log(lmax)-vcut/c] |= flag.out

    sj = slice(*np.searchsorted(lnwave_j_full, np.log([lmin, lmax])))
    lnwave_j = lnwave_j_full[sj]
    spec_cell_j = spec_cell_j_full[sj]

    ibeg, iend = np.where(flag_obs==0)[0][[0, -1]]

    len_ch = int((iend-ibeg)/chunks)
    ibeg = ibeg + chunk*len_ch
    iend = ibeg + len_ch
    if chunks > 1:
        flag_obs[:ibeg] |= flag.chunk
        flag_obs[iend:] |= flag.chunk

    if flagfile:
        for msk_i in msk_o[msk_o.order==order]:
            flag_obs[int(msk_i.start):int(msk_i.end)] |= flag.clip
        if len(msk_l):
            msk_wave = lambda x: np.interp(x, msk_w, msk_f)
            flag_obs[msk_wave(wave_obs) > 0.1] |= flag.clip

    if 1:
        kap = 6
        p17, smod, p83 = np.percentile(spec_obs[flag_obs==0], [17, 50, 83])
        sig = (p83 - p17) / 2
        flag_obs[spec_obs > smod+kap*sig] |= flag.clip

    i_ok = np.where(flag_obs==0)[0]
    pixel_ok = pixel[i_ok]
    wave_obs_ok = wave_obs[i_ok]
    spec_obs_ok = spec_obs[i_ok]

    modset['xcen'] = xcen = np.nanmean(pixel_ok) + 18
    modset['IP_hs'] = iphs

    if deg_norm_rat:
        modset['func_norm'] = lambda x, par_norm: pade(x, par_norm[:deg_norm+1], par_norm[deg_norm+1:])

    specs_molec = []
    par_atm = parfix_atm = []
    if 'add' in telluric:
        specs_molec = np.zeros((0, len(lnwave_j)))
        for mol in specs_molec_all.keys():
            s_mol = slice(*np.searchsorted(wave_atm_all[mol], [lmin, lmax]))
            if specs_molec_all[mol][s_mol] != []:
                spec_mol = np.interp(lnwave_j, np.log(wave_atm_all[mol][s_mol]), specs_molec_all[mol][s_mol])
                specs_molec = np.r_[specs_molec, [spec_mol]]
                if np.nanstd(spec_mol) > 0.0001:
                    par_atm.append((1, np.inf))
                else:
                    par_atm.append((np.nan, 0))
            else:
                specs_molec = np.r_[specs_molec, [lnwave_j*0+1]]
                par_atm.append((np.nan, 0))

        if telluric == 'add2' and len(molec) > 1:
            par_atm = np.asarray(par_atm)
            is_H2O = np.asarray(molec) == 'H2O'

            if any(is_H2O):
                specs_molec = [specs_molec[is_H2O][0], np.nanprod(specs_molec[~is_H2O]*(par_atm[~is_H2O][:, 0]).reshape(-1, 1), axis=0)]
                par_atm = [(1, np.inf), (1, np.inf)]
            else:
                specs_molec = np.nanprod(specs_molec[~is_H2O]*(par_atm[~is_H2O][:, 0]).reshape(-1, 1), axis=0)
                par_atm = [(1, np.inf)]

        if tellshift:
            par_atm.append((1, np.inf))

    if tplname:
        S_star = lambda x: np.interp(x, np.log(wave_tpl[order]) - np.log(1+berv/c), spec_tpl[order])
    else:
        S_star = lambda x: 0*x + 1

    IP_func = IPs[ip]

    S_mod = model(S_star, lnwave_j, spec_cell_j, specs_molec, IP_func, **modset)

    par = Params()
    par.rv = rv_guess if tplname else (0, 0)

    norm_guess = np.nanmean(spec_obs_ok) / np.nanmean(S_star(np.log(wave_obs_ok))) / np.nanmean(spec_cell_j)
    par.norm = [norm_guess] + [0]*deg_norm

    if deg_norm_rat:
        par.norm += [5e-7] * deg_norm_rat

    par.wave = np.polyfit(pixel_ok-xcen, wave_obs_ok, deg_wave)[::-1]
    parguess = Params(par)

    par.ip = [ip_guess['s']]
    par.atm = par_atm

    if deg_bkg:
        par.bkg = [0]

    if ip in ip_guess:
        par.ip = ip_guess[ip]
    elif ip in ('sg', 'mg', 'asg'):
        par.ip += [2.]
    elif ip in ('ag', 'agr', 'asg'):
        par.ip += [1.]
    elif ip in ('bg',):
        par.ip += [par.ip[-1]]
    parguess.ip = par.ip

    sig = 1 * err_obs if (wgt in 'error') else np.ones_like(spec_obs)
    if telluric in ('add', 'add2'):
        sig[mskatm(wave_obs) < 0.1] = tsig

    fixed = lambda x: [(pk, 0) for pk in x]

    if ip in ('sg', 'ag', 'agr', 'bg', 'bnd'):
        S_modg = model(S_star, lnwave_j, spec_cell_j, specs_molec, IPs['g'], **modset)
        par1 = Params(par, ip=par.ip[0:1])
        par2, _ = S_modg.fit(pixel_ok, spec_obs_ok, par1, sig=sig[i_ok])
        par = par + par2.flat()
    par3 = par

    if kapsig[0]:
        smod = S_mod(pixel, **par3)
        resid = spec_obs - smod
        resid[flag_obs != 0] = np.nan
        flag_obs[abs(resid) >= (kapsig[0]*np.nanstd(resid))] |= flag.clip
        i_ok = np.where(flag_obs == 0)[0]
        pixel_ok = pixel[i_ok]
        wave_obs_ok = wave_obs[i_ok]
        spec_obs_ok = spec_obs[i_ok]

    if IP_func == 'bnd':
        S_mod = model_bnd(S_star, lnwave_j, spec_cell_j, params[2], **modset)
        opt = {'x': pixel_ok, 'sig_k': par_ip[0]/1.5/c}
        rr = S_mod.fit(spec_obs_ok, 0.1, **opt)
        fx = S_mod(pixel_ok, 0.1, rr[0])
        ipxj = S_mod.IPxj(rr[0])

        e_v = np.nan
        if tplname:
            vv = np.arange(-1, 1, 0.1)
            RR = []
            aa = []
            for v in vv:
                rr = S_mod.fit(spec_obs_ok, v, **opt)
                RR.append(*rr[1])
                aa.append(rr[0])
                if 1:
                    print(v, rr[1])

            par_rv, e_v, a = SSRstat(vv, RR, plot=1, N=spec_obs_ok.size)

        best = S_mod.fit(spec_obs_ok, par_rv, **opt)
        fx = S_mod(pixel_ok, par_rv, best[0])
        S_mod.show([par_rv, best[0]], pixel_ok, spec_obs_ok, x2=pixel_ok)
        res = spec_obs_ok - fx
        np.savetxt('res.dat', list(zip(pixel_ok, res)), fmt="%s")
        prms = np.nanstd(res) / fx.nanmean() * 100
        return par_rv*1000, e_v*1000, bjd.jd, berv, best[0], np.diag(np.nan*best[0]), prms

    show = (order in look) or (order in lookfast)

    if 1:
        par.wave = parguess.wave
        if 'wave' in fix: par.wave = fixed(parguess.wave)
        if ipB:
            par.bkg = [(0, 0)]
            par.ipB = [(ipB[0], 0)]
        if deg_bkg:
            par.bkg = [0]

        par4, e_params = S_mod.fit(pixel_ok, spec_obs_ok, par, dx=0.1*show, sig=sig[i_ok], res=(not createtpl)*show, rel_fac=createtpl*show)
        par = par4

    if kapsig[-1]:
        smod = S_mod(pixel, **par)
        resid = spec_obs - smod
        resid[flag_obs != 0] = np.nan

        nr_k1 = np.count_nonzero(flag_obs)
        flag_obs[abs(resid) >= (kapsig[-1]*np.nanstd(resid))] |= flag.clip
        nr_k2 = np.count_nonzero(flag_obs)

        if nr_k1 != nr_k2:
            i_ok = np.where(flag_obs == 0)[0]
            pixel_ok = pixel[i_ok]
            wave_obs_ok = wave_obs[i_ok]
            spec_obs_ok = spec_obs[i_ok]

        if wgt in 'tell':
            atm_ok = np.array([0 if (d.unc>20 or d.unc==0) else 1 for d in par.atm])
            atm_ok[np.array(par.atm)<0.2] = 0

            if np.sum(atm_ok) > 0:
                sig = smod**2/spec_obs
                sig /= np.nanmedian(sig[i_ok])
                sig[spec_obs/np.nanmedian(spec_obs[i_ok])<0.1] = 2

        if (nr_k1 != nr_k2) or ('tell' in wgt):
            par5, e_params = S_mod.fit(pixel_ok, spec_obs_ok, par3, dx=0.1*show, sig=sig[i_ok], res=(not createtpl)*show, rel_fac=createtpl*show)
            par = par5

        if wgt in 'tell':
           sig = smod**2/spec_obs
           sig /= np.nanmedian(sig[i_ok])
           sig[spec_obs/np.nanmedian(spec_obs[i_ok])<0.1] = 2

        if (nr_k1 != nr_k2) or ('tell' in wgt):
            par5, e_params = S_mod.fit(pixel_ok, spec_obs_ok, par3, dx=0.1*show, sig=sig[i_ok], res=(not createtpl)*show, rel_fac=createtpl*show)
            par = par5

    if createtpl:
        if tplname:
            S_star = lambda x: 0*x + 1
            S_mod = model(S_star, lnwave_j, spec_cell_j, specs_molec, IP_func, **modset)

        gas_model = np.nan * np.empty_like(pixel)
        gas_model[iset] = S_mod(pixel[iset], **par)
        gas_model /= np.nanmedian(gas_model[iset])

        spec_cor = spec_obs / gas_model
        err_cor = err_obs / gas_model

        spec_cor[gas_model<0.2] = np.nan
        spec_cor[spec_cor<0.01] = np.nan

        if tpl_wave in ('initial', 'berv'):
            wave_model= wave_obs + 0
        elif tpl_wave in ('tell'):
            wave_model = np.poly1d(par.wave[::-1])(pixel-xcen)
        bervt = berv + 0
        if tpl_wave in ('initial'):
            bervt = 0

        spec_cor = np.interp(wave_model, wave_model*(1+bervt/c)/(1+par.rv/c*int(not tpl_noRV)), spec_cor/np.nanmedian(spec_cor))
        spec_cor /= np.nanmedian(spec_cor)

        weight = gas_model / (err_cor/np.nanmedian(spec_cor))**2
        weight = np.interp(wave_model, wave_model*(1+bervt/c)/(1+par.rv/c*int(not tpl_noRV)), weight)

        spec_all[order, 0][n] = wave_model
        spec_all[order, 1][n] = spec_cor
        spec_all[order, 2][n] = weight

    if show:
        gplot+(pixel[flag_obs != 0], wave_obs[flag_obs != 0], spec_obs[flag_obs != 0], 1*(flag_obs[flag_obs != 0] == flag.clip), 'us (lam?$2:$1):3:(int($4)?5:9) w p pt 6 ps 0.5 lc 9 t "flagged and clipped"')

    rvo, e_rvo = 1000*par.rv, 1000*par.rv.unc

    fmod = S_mod(pixel_ok, **par)
    res = spec_obs_ok - fmod
    prms = np.nanstd(res) / np.nanmean(fmod) * 100
    np.savetxt('res.dat', list(zip(pixel_ok, res)), fmt="%s")

    return rvo, e_rvo, bjd.jd, berv, par, e_params, prms


obsnames = np.array(sorted(glob.glob(obspath)))[nset]
obsnames = [x for x in obsnames if not any(pat in os.path.basename(x) for pat in nexcl)]

N = len(obsnames)
if not N: raise SystemExit('no files: ' + obspath)

orders = np.r_[oset]
print(orders)

rv = np.nan * np.empty(chunks*len(orders))
e_rv = np.nan * np.empty(chunks*len(orders))

rvounit = open(tag+'.rvo.dat', 'w')
parunit = open(tag+'.par.dat', 'w')

colnums = orders if chunks == 1 else [f'{order}-{ch}' for order in orders for ch in range(chunks)]
print('BJD RV e_RV BERV', *map("rv{0} e_rv{0}".format, colnums), 'filename', file=rvounit)

pixel, wave0, spec0, err0, flag0, bjd, berv = Spectrum(obsnames[0], order=orders[0])
pixel, wave1, spec1, err1, flag1, bjd, berv = Spectrum(obsnames[0], order=orders[-1])

obs_lmin = np.min([wave0[0], wave0[-1], wave1[0], wave1[-1]])
obs_lmax = np.max([wave0[0], wave0[-1], wave1[0], wave1[-1]])

####  FTS  ####
if ftsname != 'None':
    wave_cell, spec_cell, lnwave_j_full, spec_cell_j_full = FTS(ftsname)
else:
    wave_cell = np.linspace(obs_lmin, obs_lmax, len(pixel)*len(orders)*200)
    spec_cell = wave_cell*0 + 1
    lnw = np.log(wave_cell)
    lnwave_j_full = np.arange(lnw[0], lnw[-1], 200/3e8)
    spec_cell_j_full = lnwave_j_full*0 + 1

if nocell:
    spec_cell = spec_cell*0 + 1
    spec_cell_j_full = spec_cell_j_full*0 + 1

mskatm = lambda x: np.interp(x, *np.genfromtxt(viperdir+'lib/mask_vis1.0.dat').T)

if flagfile:
    msk = np.genfromtxt(flagfile, names=True, invalid_raise=False, missing_values = {'order':"-"}, filling_values={'order':np.nan}, delimiter=' ').view(np.recarray)
    msk_o = msk[np.isfinite(msk.order)]
    msk_l = msk[np.isnan(msk.order)]

    if len(msk_l):
        msk_w = np.asarray(np.concatenate([msk_l.start, msk_l.end, msk_l.start-0.05, msk_l.end+0.05]))
        msk_f = np.asarray(np.concatenate([np.ones(2*len(msk_l.start)), np.zeros(2*len(msk_l.start))]))
        ind = np.argsort(msk_w)
        msk_w, msk_f = msk_w[ind], msk_f[ind]


#### Telluric model ####
if 'add' in telluric:
    bands_all = ['vis', 'J', 'H', 'K']
    wave_band = [0, 9000, 14000, 18500]

    w0 = obs_lmin - wave_band
    w1 = obs_lmax - wave_band
    bands = bands_all[np.argmin(w0[w0 >= 0]): int(np.argmin(w1[w1 >= 0]) + 1)]

    specs_molec_all = defaultdict(list)
    wave_atm_all = defaultdict(list)
    molec_sel = molec

    for band in bands:
        hdu = fits.open(viperdir+'/lib/atmos/stdAtmos_'+band+'.fits')
        cols = hdu[1].columns.names
        data = hdu[1].data

        if molec_sel[0] == 'all': molec = cols[1:]

        for i_mol, mol in enumerate(molec):
            if (mol != 'lambda') and (mol in cols):
                specs_molec_all[mol].extend(data[mol])
                wave_atm_all[mol].extend(data['lambda'] * (1 + (-0.249/3e5)))

        molec = np.array(list(specs_molec_all.keys()))

spec_all = defaultdict(dict)

####  stellar template  ####
if tplname:
    print('reading stellar template')
    wave_tpl, spec_tpl = {}, {}
    for order in orders:
        wave_tplo, spec_tplo = Tpl(tplname, order=order)
        if oversampling:
            us = np.linspace(np.log(wave_tplo[0]), np.log(wave_tplo[-1]), oversampling*wave_tplo.size)
            spec_tplo = np.nan_to_num(spec_tplo)
            fs = CubicSpline(np.log(wave_tplo), spec_tplo)(us)
            wave_tpl[order], spec_tpl[order] = np.exp(us), fs
        else:
            wave_tpl[order], spec_tpl[order] = wave_tplo, spec_tplo
else:
    wave_tpl, spec_tpl = [wave_cell[[0, -1]]]*200, [np.ones(2)]*200


if createtpl:
    wave_tplo, spec_tplo = Tpl(obsnames[-1], order=orders[-1])
    wmax = np.max(wave_tplo)
else:
    wmax = np.max(wave_tpl[orders[-1]])

if telluric == 'add' and (wave_cell[-1] < wmax):
    wave_cell_ext = np.arange(wave_cell[-1], wmax, wave_cell[-1]-wave_cell[-2])[1:]
    spec_cell_ext = np.ones_like(wave_cell_ext)

    wave_cell = np.append(wave_cell, wave_cell_ext)
    spec_cell = np.append(spec_cell, spec_cell_ext)

    lnwave_j = np.log(wave_cell)
    lnwave_j_full = np.arange(lnwave_j[0], lnwave_j[-1], 200/3e8)
    spec_cell_j_full = np.interp(lnwave_j_full, lnwave_j, spec_cell)

    if not tplname:
        wave_tpl, spec_tpl = [wave_cell[[0, -1]]]*200, [np.ones(2)]*200

T = time.time()
headrow = True
for n, obsname in enumerate(obsnames):
    if n == 0:
        if os.path.isdir(viperdir+'res') and os.listdir(viperdir+'res'):
            os.system('rm -rf '+viperdir+'res/*.dat')
    filename = os.path.basename(obsname)
    print(f"{n+1:3d}/{N}", filename)
    for i_o, o in enumerate(orders):
        for ch in np.arange(chunks):
            try:
                gplot.RV2title = lambda x: gplot.key('title noenhanced "%s (n=%s, o=%s%s)"'% (filename, n+1, o, x))
                gplot.RV2title('')

                rv[i_o*chunks+ch], e_rv[i_o*chunks+ch], bjd, berv, params, e_params, prms = fit_chunk(o, ch, obsname=obsname)

                print(n+1, o, ch, rv[i_o*chunks+ch], e_rv[i_o*chunks+ch])
                if 'ipB' in params: params.pop('ipB')
                if not deg_bkg: params.pop('bkg', None)
                params.rv.value *= 1000.
                params.rv.unc *= 1000.

                if headrow:
                    headrow = False
                    colnames = ["".join(map(str,x)) for x in params.flat().keys()]
                    print('BJD n order chunk', *map("{0} e_{0}".format, colnames), 'prms', file=parunit)

                flat_params = [f"{d.value} {d.unc}" for d in params.flat().values()]
                print(bjd, n+1, o, ch, *flat_params, prms, file=parunit)
                os.system('mkdir -p res; touch res.dat')
                os.system('mv res.dat res/%03d_%03d.dat' % (n, o))

            except Exception as e:
                if repr(e) == 'BdbQuit()':
                    exit()
                print("Order failed due to:", repr(e))

    if not np.isnan(rv).all():
        oo = np.isfinite(e_rv)
        if oo.sum() == 1:
            RV = rv[oo][0]
            e_RV = e_rv[oo][0]
        else:
            RV = np.nanmean(rv[oo])
            e_RV = np.nanstd(rv[oo])/(oo.sum()-1)**0.5
        print('RV:', RV, e_RV, bjd, berv)

        print(bjd, RV, e_RV, berv, *sum(zip(rv, e_rv), ()), filename, file=rvounit)
        print(file=parunit)


if createtpl:
    wave_tpl_new = {}
    spec_tpl_new = {}
    err_tpl_new = {}
    orders_ok = sorted(set([kk[0] for kk in spec_all.keys()]))
    for order in orders_ok:
        gplot.reset()
        gplot.key("title 'order: %s' noenhance" % (order))
        gplot.xlabel('"Vacuum wavelength [Å]"')
        gplot.ylabel('"flux"')
        gplot.yrange("[%g:%g]" % (-1, 1.6))
        wave_t = np.array(list(spec_all[order, 0].values()))
        spec_t = np.array(list(spec_all[order, 1].values()))
        weight_t = np.array(list(spec_all[order, 2].values()))
        weight_t[np.isnan(spec_t)] = 0
        weight_t[spec_t<0] = 0

        wave_tpl_new[order] = wave_t[0]

        if len(spec_t) > 1:
            for nn in range(1, len(spec_t)):
                valid = np.isfinite(spec_t[nn])
                spec_cubic = CubicSpline(wave_t[nn][valid], spec_t[nn][valid])(wave_t[0])
                spec_cubic[valid==0] = np.nan
                spec_t[nn] = spec_cubic
                weight_t[nn] = np.interp(wave_t[0], wave_t[nn][valid], weight_t[nn][valid])
                weight_t[nn][valid==0] = np.nan
                weight_t[nn][weight_t[nn]==0] = np.nan

            if kapsig_ctpl:
                spec_mean = np.nanmedian(spec_t, axis=0)
                for nn in range(0, len(spec_t)):
                    weight_t[nn][np.abs(spec_t[nn]-spec_mean)>kapsig_ctpl] = np.nan

            spec_tpl_new[order] = np.nansum(spec_t*weight_t, axis=0) / np.nansum(weight_t, axis=0)
            err_tpl_new[order] = np.nanstd(spec_t, axis=0)

        else:
            spec_tpl_new[order] = spec_t[0]
            err_tpl_new[order] = spec_t[0]*np.nan

        if (order in lookfast) or (order in look):
            gplot(wave_tpl_new[order], spec_tpl_new[order] - 1 , 'w l lc 7 t "combined tpl"')
            for n in range(len(spec_t)):
                gplot+(wave_tpl_new[order], spec_t[n]/np.nanmedian(spec_t[n]), 'w l t "%s"' % (os.path.split(obsnames[n])[1]))

    write_fits(wave_tpl_new, spec_tpl_new, err_tpl_new, obsnames, tag)

rvounit.close()
parunit.close()

T = time.time() - T
Tfmt = lambda t: time.strftime("%Hh%Mm%Ss", time.gmtime(t))
print("processing time total:       ", Tfmt(T))
print("processing time per spectrum:", Tfmt(T/N))
print("processing time per chunk:   ", Tfmt(T/N/orders.size))

tag += '.rvo.dat'

print(tag, 'done.')

def run():
    pass
