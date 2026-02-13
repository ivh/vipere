#! /usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "astropy",
#   "matplotlib",
#   "numpy",
#   "pyyaml",
#   "scipy",
# ]
# ///
## Licensed under a GPLv3 style license - see LICENSE
## vipere - Telluric correction for CRIRES+ spectra
## Author: Alexis Lavail (with help from Claude)
## Forked and adapted from viper (https://github.com/mzechmeister/viper)
## Original authors: Mathias Zechmeister and Jana Koehler

import argparse
import glob
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit, least_squares
from scipy.sparse import lil_matrix
from scipy.special import erf
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt

plt.ion()

c = 299792.458   # [km/s] speed of light
viperdir = os.path.dirname(os.path.realpath(__file__)) + os.sep


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

IPs = {'g': IP, 'sg': IP_sg, 'ag': IP_ag, 'agr': IP_agr, 'asg': IP_asg, 'bg': IP_bg, 'mg': IP_mg, 'mcg': IP_mcg, 'lor': IP_lor}


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
        self.S_star, self.lnwave_j, self.fluxes_molec, self.IP = args
        self.dx = self.lnwave_j[1] - self.lnwave_j[0]
        self.IP_hs = IP_hs
        self.vk = np.arange(-IP_hs, IP_hs+1) * self.dx * c
        self.lnwave_j_eff = self.lnwave_j[IP_hs:-IP_hs]
        self.func_norm = func_norm

    def __call__(self, pixel, rv=0, norm=[1], wave=[], ip=[], atm=[], bkg=[0], ipB=[]):
        coeff_norm, coeff_wave, coeff_ip, coeff_atm, coeff_bkg, coeff_ipB = norm, wave, ip, atm, bkg, ipB

        spec_gas = 1

        if len(self.fluxes_molec):
            flux_atm = np.nanprod(np.power(self.fluxes_molec, np.abs(coeff_atm[:len(self.fluxes_molec)])[:, np.newaxis]), axis=0)
            if len(coeff_atm) == len(self.fluxes_molec)+1:
                flux_atm = np.interp(self.lnwave_j, self.lnwave_j-np.log(1+coeff_atm[-1]/c), flux_atm)
            spec_gas = flux_atm

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
        prms = np.nan

        fig = plt.figure(1)
        fig.clf()

        title = getattr(fig, '_rv2title', '')
        if par_rv:
            title += ", v=%.2f \u00b1 %.2f m/s" % (par_rv*1000, par_rv.unc*1000)

        show_stellar = res and len(self.fluxes_molec)

        if res or rel_fac:
            col2 = rel_fac * np.mean(ymod) * (y/ymod - 1) if rel_fac else y - ymod
            rms = np.std(col2)
            prms = rms / np.mean(ymod) * 100
            if show_stellar:
                ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
            else:
                ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                ax3 = None
        else:
            ax1 = fig.subplots(1, 1)
            ax2 = None
            ax3 = None

        ax1.plot(x2, y, '.', ms=3, label='obs')
        ax1.plot(x2, ymod, '.', ms=3, color='C2', label='model')
        if dx:
            xx = np.arange(x.min(), x.max(), dx)
            xx2 = np.poly1d(params.wave[::-1])(xx-self.xcen)
            yymod = self(xx, **params)
            ax1.plot(xx2, yymod, '-', color='C2', lw=0.8)
        ax1.set_ylim(top=1.4*np.nanmax(ymod))
        ax1.set_ylabel('flux')
        ax1.legend(loc='upper right', fontsize='small')
        if title:
            ax1.set_title(title, fontsize='small')

        if ax2 is not None:
            style = '.' if res else '-'
            ax2.plot(x2, col2, style, ms=3, color='C0', label='res (%.3g ~ %.3g%%)' % (rms, prms))
            ax2.axhline(0, color='C2', lw=0.8)
            ax2.set_ylabel('residuals')
            ax2.legend(loc='upper right', fontsize='small')

        if ax3 is not None:
            orig_S_star = self.S_star
            self.S_star = lambda x: np.ones_like(x)
            ygas = self(x, **params)
            self.S_star = orig_S_star
            stellar = y / ygas
            stellar /= np.nanmedian(stellar)
            ax3.plot(x2, stellar, '.', ms=3, color='C0')
            ax3.set_xlabel(u'Vacuum wavelength [\u00c5]')
            ax3.set_ylabel('stellar')
        elif ax2 is not None:
            ax2.set_xlabel(u'Vacuum wavelength [\u00c5]')
        else:
            ax1.set_xlabel(u'Vacuum wavelength [\u00c5]')

        fig.tight_layout()
        plt.pause(0.01)
        return prms


###############################################################################
# CRIRES instrument
###############################################################################

crires_location = EarthLocation.from_geodetic(
    lat=-24.6268 * u.deg, lon=-70.4045 * u.deg, height=2648 * u.m
)

oset = '1:28'  # covers up to 9 orders/det (Y/J band); K/H use fewer via -oset
ip_guess = {'s': 1.5}


def Spectrum(filename='', order=None, targ=None):

    order_idx, detector = divmod(order-1, 3)
    detector += 1

    exptime = 0

    hdu = fits.open(filename, ignore_blank=True)
    hdr = hdu[0].header
    ra = hdr.get('RA', np.nan)
    de = hdr.get('DEC', np.nan)
    nod_type = hdr['ESO PRO CATG']


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

    wave = (hdu[detector].data["0"+str(order_drs)+"_01_WL"]) * 10

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
    out=      2,
    clip=     4,
    chunk=    8,
)


def local_sigma(resid, halfwin=50):
    """Sliding-window MAD-based sigma estimate (scaled to Gaussian std)."""
    n = len(resid)
    sigma = np.empty(n)
    for i in range(n):
        lo = max(0, i - halfwin)
        hi = min(n, i + halfwin + 1)
        window = resid[lo:hi]
        valid = window[np.isfinite(window)]
        sigma[i] = np.nanmedian(np.abs(valid - np.nanmedian(valid))) * 1.4826 if len(valid) > 3 else np.inf
    return sigma


def arg2slice(arg):
    """Convert string argument to a slice."""
    if isinstance(arg, str):
        arg = eval('np.s_['+arg+']')
    return [arg] if isinstance(arg, int) else arg

def arg2range(arg):
    return  eval('np.r_['+arg+']')


if __name__ == "__main__" or __name__ == "vipere":
    argparse.ArgumentDefaultsHelpFormatter._split_lines = lambda self, text, width: text.splitlines()

    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument('args', nargs='*')
    preparser.add_argument('-inst', help='Instrument.', default='CRIRES', choices=insts)
    preparser.add_argument('-config_file', help='YAML config file to override defaults.', type=str)
    preargs = preparser.parse_known_args()[0]

    iset = slice(None)

    # read in default values from config_vipere.yaml
    import yaml
    with open(viperdir+'config_vipere.yaml') as f:
        configs_def = {k: str(v) for k, v in yaml.safe_load(f).items()}

    configs_user = {}
    if preargs.config_file:
        with open(preargs.config_file[0]) as f:
            configs_user = {k: str(v) for k, v in yaml.safe_load(f).items()}

    parser = argparse.ArgumentParser(description='vipere - Telluric correction for CRIRES+ spectra', add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argopt = parser.add_argument
    argopt('obspath', help='Filename of observation.', default='data/TLS/betgem/BETA_GEM.fits', type=str)
    argopt('tplname', help='Filename of template.', nargs='?', type=str)
    argopt('-inst', help='Instrument.', default='CRIRES', choices=insts)
    argopt('-ip', help='IP model (g: Gaussian, ag: asymmetric (skewed) Gaussian, sg: super Gaussian, bg: biGaussian, mg: multiple Gaussians, mcg: multiple central Gaussians).', default='g', choices=[*IPs], type=str)
    argopt('-chunks', nargs='?', help='Divide one order into a number of chunks.', default=1, type=int)
    argopt('-config_file', help='YAML config file to override defaults.', type=str)
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
    argopt('-plot', help='Plot level: 0=off, 1=plot with pause, 2=plot without pause.', default=0, type=int)
    argopt('-molec', nargs='*', help='Molecular specifies; all: Automatic selection of all present molecules.', default=['all'], type=str)
    argopt('-nexcl', nargs='*', help='Ignore spectra with string pattern.', default=[], type=str)
    argopt('-nset', help='Index for spectrum.', default=':', type=arg2slice)
    argopt('-oset', help='Index for order.', default=oset, type=arg2slice)
    argopt('-oversampling', help='Oversampling factor for the template data.', default=None, type=int)
    argopt('-rv_guess', help='RV guess.', default=1., type=float)
    argopt('-o', dest='tag', help='Output basename for result files.', default='tmp', type=str)
    argopt('-tellshift', nargs='?', help='Variable telluric wavelength shift (one value for all selected molecules).', default=False, const=True, type=int)
    argopt('-tell_bic', help='BIC threshold for telluric model selection. The telluric model must improve BIC by at least this amount to be preferred over the no-telluric model. Set to 0 to disable.', default=10, type=float)
    argopt('-telluric', help='Treating tellurics (add: telluric forward modelling with one coeff for each molecule; add2: telluric forward modelling with combined coeff for non-water molecules).', default='', choices=['', 'add', 'add2'], type=str)
    argopt('-global_atm', help='Fit atmosphere globally across all orders.', default=False, action='store_true')
    argopt('-tpl_noRV', nargs='?', help='No stellar RV shift is applied to the telluric corrected spectrum. Just in combination with -createtpl.', default=False, const=True, type=int)
    argopt('-tpl_wave', help='Output wavelength of generated template (initial: take wavelengths from imput file; berv: apply barycentric correction to input wavelengths; tell: updated wavelength solution estimated via telluric lines).', default='initial', type=str)
    argopt('-tsig', help='(Relative) sigma value for weighting tellurics.', default=1, type=float)
    argopt('-vcut', help='Trim the observation to a range valid for the model [km/s]', default=100, type=float)
    argopt('-wgt', nargs='?', help='Weighted least square fit (error: employ data error; tell: upweight tellurics and downweight stellar lines)', default='', type=str)
    argopt('-?', '-h', '-help', '--help', help='Show this help message and exit.', action='help')

    parser.set_defaults(**configs_def)
    parser.set_defaults(**configs_user)

    parser.set_defaults(kapsig = [float(i) for i in (argopt('--kapsig').default.split(' '))])

    args = parser.parse_args()
    globals().update(vars(args))


def setup_chunk(order, chunk, obsname, rv_prev=None):
    '''Set up data, model and parameters for one order/chunk. Returns a dict.'''
    ####  observation  ####
    pixel, wave_obs, spec_obs, err_obs, flag_obs, bjd, berv = Spectrum(obsname, order=order)

    flag_obs[np.isnan(spec_obs)] |= flag.nan

    lmin = max(wave_obs[iset][0], wave_tpl[order][0], wave_grid[0])
    lmax = min(wave_obs[iset][-1], wave_tpl[order][-1], wave_grid[-1])

    flag_obs[np.log(wave_obs) < np.log(lmin)+vcut/c] |= flag.out
    flag_obs[np.log(wave_obs) > np.log(lmax)-vcut/c] |= flag.out

    sj = slice(*np.searchsorted(lnwave_j_full, np.log([lmin, lmax])))
    lnwave_j = lnwave_j_full[sj]

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

    kap = 6
    p17, smod, p83 = np.percentile(spec_obs[flag_obs==0], [17, 50, 83])
    sig = (p83 - p17) / 2
    flag_obs[spec_obs > smod+kap*sig] |= flag.clip

    i_ok = np.where(flag_obs==0)[0]
    pixel_ok = pixel[i_ok]
    wave_obs_ok = wave_obs[i_ok]
    spec_obs_ok = spec_obs[i_ok]

    modset_local = dict(modset)
    modset_local['xcen'] = xcen = np.nanmean(pixel_ok) + 18
    modset_local['IP_hs'] = iphs

    if deg_norm_rat:
        modset_local['func_norm'] = lambda x, par_norm: pade(x, par_norm[:deg_norm+1], par_norm[deg_norm+1:])

    specs_molec = []
    par_atm = []
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

    S_mod = model(S_star, lnwave_j, specs_molec, IP_func, **modset_local)

    par = Params()
    if not tplname:
        par.rv = (0, 0)
    elif rv_prev is not None:
        par.rv = rv_prev
    else:
        par.rv = rv_guess

    norm_guess = np.nanmean(spec_obs_ok) / np.nanmean(S_star(np.log(wave_obs_ok)))
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

    sig = 1 * err_obs if (wgt == 'error') else np.ones_like(spec_obs)
    if telluric in ('add', 'add2'):
        sig[mskatm(wave_obs) < 0.1] = tsig

    if ip in ('sg', 'ag', 'agr', 'bg'):
        S_modg = model(S_star, lnwave_j, specs_molec, IPs['g'], **modset_local)
        par1 = Params(par, ip=par.ip[0:1])
        par2, _ = S_modg.fit(pixel_ok, spec_obs_ok, par1, sig=sig[i_ok])
        par = par + par2.flat()
    par3 = par

    if kapsig[0]:
        smod = S_mod(pixel, **par3)
        resid = spec_obs - smod
        resid[flag_obs != 0] = np.nan
        flag_obs[abs(resid) >= kapsig[0] * local_sigma(resid)] |= flag.clip
        i_ok = np.where(flag_obs == 0)[0]
        pixel_ok = pixel[i_ok]
        wave_obs_ok = wave_obs[i_ok]
        spec_obs_ok = spec_obs[i_ok]

    return {
        'pixel': pixel, 'wave_obs': wave_obs, 'spec_obs': spec_obs,
        'err_obs': err_obs, 'flag_obs': flag_obs, 'bjd': bjd, 'berv': berv,
        'i_ok': i_ok, 'pixel_ok': pixel_ok, 'wave_obs_ok': wave_obs_ok,
        'spec_obs_ok': spec_obs_ok,
        'S_mod': S_mod, 'S_star': S_star, 'IP_func': IP_func,
        'specs_molec': specs_molec, 'lnwave_j': lnwave_j,
        'par': par, 'par3': par3, 'parguess': parguess, 'par_atm': par_atm,
        'sig': sig, 'xcen': xcen, 'modset': modset_local,
        'order': order, 'chunk': chunk,
    }


def fit_chunk(order, chunk, obsname, rv_prev=None):
    s = setup_chunk(order, chunk, obsname, rv_prev)
    pixel = s['pixel']; wave_obs = s['wave_obs']; spec_obs = s['spec_obs']
    err_obs = s['err_obs']; flag_obs = s['flag_obs']; bjd = s['bjd']; berv = s['berv']
    i_ok = s['i_ok']; pixel_ok = s['pixel_ok']; wave_obs_ok = s['wave_obs_ok']
    spec_obs_ok = s['spec_obs_ok']
    S_mod = s['S_mod']; S_star = s['S_star']; IP_func = s['IP_func']
    specs_molec = s['specs_molec']; lnwave_j = s['lnwave_j']
    par = s['par']; par3 = s['par3']; parguess = s['parguess']; par_atm = s['par_atm']
    sig = s['sig']; xcen = s['xcen']; modset_local = s['modset']

    fixed = lambda x: [(pk, 0) for pk in x]
    show = plot > 0

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
        flag_obs[abs(resid) >= kapsig[-1] * local_sigma(resid)] |= flag.clip
        nr_k2 = np.count_nonzero(flag_obs)

        if nr_k1 != nr_k2:
            i_ok = np.where(flag_obs == 0)[0]
            pixel_ok = pixel[i_ok]
            wave_obs_ok = wave_obs[i_ok]
            spec_obs_ok = spec_obs[i_ok]

        if wgt == 'tell':
            atm_ok = np.array([0 if (d.unc>20 or d.unc==0) else 1 for d in par.atm])
            atm_ok[np.array(par.atm)<0.2] = 0

            if np.sum(atm_ok) > 0:
                sig = smod**2/spec_obs
                sig /= np.nanmedian(sig[i_ok])
                sig[spec_obs/np.nanmedian(spec_obs[i_ok])<0.1] = 2

        if (nr_k1 != nr_k2) or (wgt == 'tell'):
            par5, e_params = S_mod.fit(pixel_ok, spec_obs_ok, par3, dx=0.1*show, sig=sig[i_ok], res=(not createtpl)*show, rel_fac=createtpl*show)
            par = par5

    # BIC model comparison: telluric vs no-telluric
    if telluric and len(specs_molec):
        fmod_tell = S_mod(pixel_ok, **par)
        rss_tell = np.sum((spec_obs_ok - fmod_tell)**2)
        k_tell = len(par.vary())
        n_data = len(pixel_ok)
        bic_tell = n_data * np.log(rss_tell / n_data) + k_tell * np.log(n_data)

        S_mod_notell = model(S_star, lnwave_j, [], IP_func, **modset_local)
        par_notell = Params(parguess)
        if 'wave' in fix:
            par_notell.wave = [(p.value, 0) for p in parguess.wave]
        if deg_bkg:
            par_notell.bkg = [0]
        if ipB:
            par_notell.bkg = [(0, 0)]
            par_notell.ipB = [(ipB[0], 0)]
        sig_notell = 1 * err_obs if (wgt == 'error') else np.ones_like(spec_obs)

        try:
            par_notell_fit, e_params_notell = S_mod_notell.fit(
                pixel_ok, spec_obs_ok, par_notell, sig=sig_notell[i_ok])
            fmod_notell = S_mod_notell(pixel_ok, **par_notell_fit)
            rss_notell = np.sum((spec_obs_ok - fmod_notell)**2)
            k_notell = len(par_notell_fit.vary())
            bic_notell = n_data * np.log(rss_notell / n_data) + k_notell * np.log(n_data)

            if bic_notell <= bic_tell + tell_bic:
                print('  BIC: no-telluric preferred (%.1f vs %.1f, threshold %.1f)' % (bic_notell, bic_tell, tell_bic))
                par_notell_fit.atm = [(np.nan, 0)] * len(par_atm)
                par = par_notell_fit
                S_mod = S_mod_notell
                e_params = e_params_notell
                if show:
                    S_mod.show(par, pixel_ok, spec_obs_ok, par_rv=par.rv,
                               dx=0.1, res=(not createtpl)*show, rel_fac=createtpl*show)
            else:
                print('  BIC: telluric preferred (%.1f vs %.1f)' % (bic_tell, bic_notell))
        except Exception:
            print('  BIC: no-telluric fit failed, keeping telluric model')

    if createtpl:
        if tplname:
            S_star = lambda x: 0*x + 1
            S_mod = model(S_star, lnwave_j, specs_molec, IP_func, **modset_local)

        gas_model = np.nan * np.empty_like(pixel)
        gas_model[iset] = S_mod(pixel[iset], **par)
        gas_model /= np.nanmedian(gas_model[iset])

        bad = (gas_model < 0.2) | np.isnan(gas_model)
        gas_model[bad] = np.nan
        spec_cor = spec_obs / gas_model
        err_cor = err_obs / gas_model

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
        fig = plt.figure(1)
        if fig.axes:
            fig.axes[0].plot(wave_obs[flag_obs != 0], spec_obs[flag_obs != 0], 'x', ms=3, color='gray', label='flagged')
            fig.axes[0].legend(loc='upper right', fontsize='small')
            plt.pause(0.01)
        if plot == 1:
            input('Press Enter to continue...')

    rvo, e_rvo = 1000*par.rv, 1000*par.rv.unc

    fmod = S_mod(pixel_ok, **par)
    res = spec_obs_ok - fmod
    prms = np.nanstd(res) / np.nanmean(fmod) * 100
    np.savetxt('res.dat', list(zip(pixel_ok, res)), fmt="%s")

    return rvo, e_rvo, bjd.jd, berv, par, e_params, prms


def _order_par(par, o):
    '''Extract shared (rv, atm) + order-specific params from combined Params.'''
    opar = Params()
    opar.rv = par.rv
    opar.atm = par.atm
    opar.norm = par['norm_o%d' % o]
    opar.wave = par['wave_o%d' % o]
    opar.ip = par['ip_o%d' % o]
    opar.bkg = par['bkg_o%d' % o]
    return opar


def _build_sparsity(varykeys, orders, boundaries, n_data):
    '''Build Jacobian sparsity matrix: shared params affect all rows,
    per-order params affect only their order's rows.'''
    n_params = len(varykeys)
    sparsity = lil_matrix((n_data, n_params), dtype=int)

    # Map order suffix to row slice
    order_slices = {o: slice(boundaries[idx], boundaries[idx+1])
                    for idx, o in enumerate(orders)}

    for j, key in enumerate(varykeys):
        # Check if this is a per-order param (contains '_o<number>')
        order_id = None
        if isinstance(key, tuple):
            name = str(key[0])
        else:
            name = str(key)
        for o in orders:
            suffix = '_o%d' % o
            if suffix in name:
                order_id = o
                break

        if order_id is not None:
            # Per-order param: only affects its order's rows
            sparsity[order_slices[order_id], j] = 1
        else:
            # Shared param (rv, atm): affects all rows
            sparsity[:, j] = 1

    return sparsity.tocsc()


def _run_least_squares(combined, orders, models, boundaries,
                       pixel_cat, spec_cat, sig_cat):
    '''Run least_squares with sparse Jacobian on the joint problem.'''
    varykeys, varyvals = zip(*combined.vary().items())
    varyvals = np.array(varyvals, dtype=float)
    n_data = len(pixel_cat)

    sparsity = _build_sparsity(varykeys, orders, boundaries, n_data)

    def residual_func(params):
        par_now = combined + dict(zip(varykeys, params))
        out = np.empty(n_data)
        for idx, o in enumerate(orders):
            opar = _order_par(par_now, o)
            sl = slice(boundaries[idx], boundaries[idx+1])
            out[sl] = (models[o](pixel_cat[sl], **opar) - spec_cat[sl]) / sig_cat[sl]
        return out

    result = least_squares(residual_func, varyvals, jac_sparsity=sparsity,
                           method='trf', x_scale='jac')

    par_fit = combined + dict(zip(varykeys, result.x))

    # Covariance: (J^T J)^{-1} * s^2, where s^2 = cost*2 / (n-p)
    J = result.jac
    n_p = n_data - len(varyvals)
    s2 = 2 * result.cost / max(n_p, 1)
    try:
        JtJ = (J.T @ J).toarray() if hasattr(J, 'toarray') else J.T @ J
        cov = np.linalg.inv(JtJ) * s2
        for k, v in zip(varykeys, np.sqrt(np.diag(cov))):
            par_fit[k].unc = v
    except np.linalg.LinAlgError:
        cov = np.full((len(varyvals), len(varyvals)), np.inf)
        for k in varykeys:
            par_fit[k].unc = np.inf

    return par_fit, cov


def fit_multi(setups, orders):
    '''Joint fit with shared rv+atm and per-order norm/wave/ip/bkg.'''
    # Build combined Params: shared rv + atm, per-order everything else
    s0 = setups[orders[0]]
    combined = Params()
    combined.rv = s0['par'].rv

    # For shared atm: a molecule is free if free in ANY order;
    # initialize value from median of per-order pre-fits (par3)
    n_atm = len(s0['par_atm'])
    shared_atm = list(s0['par_atm'])
    for o in orders[1:]:
        for i, pa in enumerate(setups[o]['par_atm']):
            if i < n_atm and pa[1] != 0:
                shared_atm[i] = pa

    # Collect pre-fitted atm values from par3 across orders
    atm_values = np.full((len(orders), n_atm), np.nan)
    for idx, o in enumerate(orders):
        par3_atm = setups[o]['par3'].atm
        for i in range(min(len(par3_atm), n_atm)):
            if shared_atm[i][1] != 0:  # only for free params
                atm_values[idx, i] = float(par3_atm[i])

    for i in range(n_atm):
        if shared_atm[i][1] != 0:
            vals = atm_values[:, i]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                shared_atm[i] = (float(np.median(vals)), np.inf)

    combined.atm = shared_atm

    for o in orders:
        s = setups[o]
        par_o = s['par']
        combined['norm_o%d' % o] = par_o.norm
        combined['wave_o%d' % o] = par_o.wave
        combined['ip_o%d' % o] = par_o.ip
        combined['bkg_o%d' % o] = par_o.bkg if 'bkg' in par_o else [0]

    # Prepare wave params (reset to parguess, apply fixes)
    fixed_fn = lambda x: [(pk, 0) for pk in x]
    for o in orders:
        s = setups[o]
        combined['wave_o%d' % o] = s['parguess'].wave
        if 'wave' in fix:
            combined['wave_o%d' % o] = fixed_fn(s['parguess'].wave)
        if ipB:
            combined['bkg_o%d' % o] = [(0, 0)]
        if deg_bkg:
            combined['bkg_o%d' % o] = [0]

    # Concatenate data across orders
    pixel_cat = np.concatenate([setups[o]['pixel_ok'] for o in orders])
    spec_cat = np.concatenate([setups[o]['spec_obs_ok'] for o in orders])
    sig_cat = np.concatenate([setups[o]['sig'][setups[o]['i_ok']] for o in orders])
    boundaries = np.cumsum([0] + [len(setups[o]['pixel_ok']) for o in orders])

    # Collect models
    models = {o: setups[o]['S_mod'] for o in orders}

    # Joint fit with sparse Jacobian
    par_fit, cov = _run_least_squares(combined, orders, models, boundaries,
                                      pixel_cat, spec_cat, sig_cat)

    # Post-clip with kapsig[-1] per order, then refit if needed
    any_clipped = False
    if kapsig[-1]:
        for idx, o in enumerate(orders):
            s = setups[o]
            opar = _order_par(par_fit, o)
            smod_full = models[o](s['pixel'], **opar)
            resid = s['spec_obs'] - smod_full
            resid[s['flag_obs'] != 0] = np.nan
            nr_k1 = np.count_nonzero(s['flag_obs'])
            s['flag_obs'][abs(resid) >= kapsig[-1] * local_sigma(resid)] |= flag.clip
            nr_k2 = np.count_nonzero(s['flag_obs'])
            if nr_k1 != nr_k2:
                any_clipped = True
                s['i_ok'] = np.where(s['flag_obs'] == 0)[0]
                s['pixel_ok'] = s['pixel'][s['i_ok']]
                s['wave_obs_ok'] = s['wave_obs'][s['i_ok']]
                s['spec_obs_ok'] = s['spec_obs'][s['i_ok']]

        if any_clipped:
            # Rebuild concatenated arrays and refit
            pixel_cat = np.concatenate([setups[o]['pixel_ok'] for o in orders])
            spec_cat = np.concatenate([setups[o]['spec_obs_ok'] for o in orders])
            sig_cat = np.concatenate([setups[o]['sig'][setups[o]['i_ok']] for o in orders])
            boundaries = np.cumsum([0] + [len(setups[o]['pixel_ok']) for o in orders])

            # Refit using par3 seeds (pre-main-fit params) with shared atm
            combined2 = Params()
            combined2.rv = s0['par3'].rv
            combined2.atm = shared_atm
            for o in orders:
                s = setups[o]
                par3_o = s['par3']
                combined2['norm_o%d' % o] = par3_o.norm
                combined2['wave_o%d' % o] = s['parguess'].wave
                if 'wave' in fix:
                    combined2['wave_o%d' % o] = fixed_fn(s['parguess'].wave)
                combined2['ip_o%d' % o] = par3_o.ip
                combined2['bkg_o%d' % o] = par3_o.bkg if 'bkg' in par3_o else [0]
                if ipB:
                    combined2['bkg_o%d' % o] = [(0, 0)]
                if deg_bkg:
                    combined2['bkg_o%d' % o] = [0]

            par_fit, cov = _run_least_squares(combined2, orders, models,
                                              boundaries, pixel_cat,
                                              spec_cat, sig_cat)

    return par_fit, cov


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

####  Log-lambda grid  ####
wave_grid = np.linspace(obs_lmin, obs_lmax, len(pixel)*len(orders)*200)
lnw = np.log(wave_grid)
lnwave_j_full = np.arange(lnw[0], lnw[-1], 200/3e8)

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
    wave_tpl, spec_tpl = [wave_grid[[0, -1]]]*200, [np.ones(2)]*200


if createtpl:
    wave_tplo, spec_tplo = Tpl(obsnames[-1], order=orders[-1])
    wmax = np.max(wave_tplo)
else:
    wmax = np.max(wave_tpl[orders[-1]])

if telluric == 'add' and (wave_grid[-1] < wmax):
    wave_grid_ext = np.arange(wave_grid[-1], wmax, wave_grid[-1]-wave_grid[-2])[1:]
    wave_grid = np.append(wave_grid, wave_grid_ext)

    lnwave_j = np.log(wave_grid)
    lnwave_j_full = np.arange(lnwave_j[0], lnwave_j[-1], 200/3e8)

    if not tplname:
        wave_tpl, spec_tpl = [wave_grid[[0, -1]]]*200, [np.ones(2)]*200

T = time.time()
headrow = True
for n, obsname in enumerate(obsnames):
    if n == 0:
        if os.path.isdir(viperdir+'res') and os.listdir(viperdir+'res'):
            os.system('rm -rf '+viperdir+'res/*.dat')
    filename = os.path.basename(obsname)
    print(f"{n+1:3d}/{N}", filename)
    rv_prev_order = None

    if global_atm and chunks == 1:
        # Phase 1: Setup all orders
        setups = {}
        for o in orders:
            try:
                setups[o] = setup_chunk(o, 0, obsname)
            except Exception as e:
                if repr(e) == 'BdbQuit()':
                    exit()
                print("Order %d setup failed: %s" % (o, repr(e)))

        if not setups:
            continue

        # Phase 2: Joint fit
        setup_orders = list(setups.keys())
        try:
            par_fit, cov_fit = fit_multi(setups, setup_orders)
        except Exception as e:
            if repr(e) == 'BdbQuit()':
                exit()
            print("Global fit failed: %s" % repr(e))
            continue

        rvo_shared = 1000 * par_fit.rv
        e_rvo_shared = 1000 * par_fit.rv.unc

        # Phase 3: Per-order post-processing
        for i_o, o in enumerate(orders):
            if o not in setups:
                continue
            s = setups[o]
            opar = _order_par(par_fit, o)
            pixel = s['pixel']; wave_obs = s['wave_obs']; spec_obs = s['spec_obs']
            err_obs = s['err_obs']; flag_obs = s['flag_obs']
            bjd = s['bjd']; berv = s['berv']
            i_ok = s['i_ok']; pixel_ok = s['pixel_ok']
            wave_obs_ok = s['wave_obs_ok']; spec_obs_ok = s['spec_obs_ok']
            S_mod_o = s['S_mod']; S_star_o = s['S_star']; IP_func_o = s['IP_func']
            specs_molec_o = s['specs_molec']; lnwave_j_o = s['lnwave_j']
            xcen_o = s['xcen']; modset_o = s['modset']

            rv[i_o] = rvo_shared
            e_rv[i_o] = e_rvo_shared

            show = plot > 0
            if show:
                fig1 = plt.figure(1)
                fig1.clf()
                fig1._rv2title = '%s (n=%s, o=%s)' % (filename, n+1, o)
                S_mod_o.show(opar, pixel_ok, spec_obs_ok, par_rv=opar.rv,
                             dx=0.1, res=(not createtpl), rel_fac=createtpl*1)

            # createtpl processing
            if createtpl:
                if tplname:
                    S_star_ct = lambda x: 0*x + 1
                    S_mod_ct = model(S_star_ct, lnwave_j_o, specs_molec_o, IP_func_o, **modset_o)
                else:
                    S_mod_ct = S_mod_o

                gas_model = np.nan * np.empty_like(pixel)
                gas_model[iset] = S_mod_ct(pixel[iset], **opar)
                gas_model /= np.nanmedian(gas_model[iset])

                bad = (gas_model < 0.2) | np.isnan(gas_model)
                gas_model[bad] = np.nan
                spec_cor = spec_obs / gas_model
                err_cor = err_obs / gas_model

                spec_cor[spec_cor<0.01] = np.nan

                if tpl_wave in ('initial', 'berv'):
                    wave_model = wave_obs + 0
                elif tpl_wave in ('tell'):
                    wave_model = np.poly1d(opar.wave[::-1])(pixel-xcen_o)
                bervt = berv + 0
                if tpl_wave in ('initial'):
                    bervt = 0

                spec_cor = np.interp(wave_model, wave_model*(1+bervt/c)/(1+opar.rv/c*int(not tpl_noRV)), spec_cor/np.nanmedian(spec_cor))
                spec_cor /= np.nanmedian(spec_cor)

                weight = gas_model / (err_cor/np.nanmedian(spec_cor))**2
                weight = np.interp(wave_model, wave_model*(1+bervt/c)/(1+opar.rv/c*int(not tpl_noRV)), weight)

                spec_all[o, 0][n] = wave_model
                spec_all[o, 1][n] = spec_cor
                spec_all[o, 2][n] = weight

            # Output
            if show:
                fig = plt.figure(1)
                if fig.axes:
                    fig.axes[0].plot(wave_obs[flag_obs != 0], spec_obs[flag_obs != 0], 'x', ms=3, color='gray', label='flagged')
                    fig.axes[0].legend(loc='upper right', fontsize='small')
                    plt.pause(0.01)
                if plot == 1:
                    input('Press Enter to continue...')

            fmod = S_mod_o(pixel_ok, **opar)
            res = spec_obs_ok - fmod
            prms = np.nanstd(res) / np.nanmean(fmod) * 100
            np.savetxt('res.dat', list(zip(pixel_ok, res)), fmt="%s")

            print(n+1, o, 0, rv[i_o], e_rv[i_o])

            # Build per-order params for .par.dat output
            params = Params(opar)
            if 'ipB' in params: params.pop('ipB')
            if not deg_bkg: params.pop('bkg', None)
            params.rv.value *= 1000.
            params.rv.unc *= 1000.

            if headrow:
                headrow = False
                colnames = ["".join(map(str,x)) for x in params.flat().keys()]
                print('BJD n order chunk', *map("{0} e_{0}".format, colnames), 'prms', file=parunit)

            flat_params = [f"{d.value} {d.unc}" for d in params.flat().values()]
            print(bjd, n+1, o, 0, *flat_params, prms, file=parunit)
            os.system('mkdir -p res; touch res.dat')
            os.system('mv res.dat res/%03d_%03d.dat' % (n, o))

            if plot > 0:
                plt.figure(1).savefig('%s_n%03d_o%03d.png' % (tag, n, o), dpi=150)

    else:
        # Existing per-order loop
        for i_o, o in enumerate(orders):
            for ch in np.arange(chunks):
                try:
                    fig1 = plt.figure(1)
                    fig1._rv2title = '%s (n=%s, o=%s)' % (filename, n+1, o)

                    rv[i_o*chunks+ch], e_rv[i_o*chunks+ch], bjd, berv, params, e_params, prms = fit_chunk(o, ch, obsname=obsname, rv_prev=rv_prev_order)

                    if np.isfinite(e_rv[i_o*chunks+ch]) and e_rv[i_o*chunks+ch] < 100:
                        rv_prev_order = rv[i_o*chunks+ch] / 1000  # m/s → km/s
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

                    if plot > 0:
                        plt.figure(1).savefig('%s_n%03d_o%03d.png' % (tag, n, o), dpi=150)

                except Exception as e:
                    if repr(e) == 'BdbQuit()':
                        exit()
                    print("Order failed due to:", repr(e))

    if not np.isnan(rv).all():
        oo = np.isfinite(e_rv) & (e_rv > 0)
        if oo.sum() == 1:
            RV = rv[oo][0]
            e_RV = e_rv[oo][0]
        elif oo.sum() > 1:
            w = 1 / e_rv[oo]**2
            RV = np.average(rv[oo], weights=w)
            e_RV = 1 / np.sqrt(np.sum(w))
        else:
            RV = np.nanmean(rv)
            e_RV = np.nan
        print('RV:', RV, e_RV, bjd, berv)

        print(bjd, RV, e_RV, berv, *sum(zip(rv, e_rv), ()), filename, file=rvounit)
        print(file=parunit)


if createtpl:
    wave_tpl_new = {}
    spec_tpl_new = {}
    err_tpl_new = {}
    orders_ok = sorted(set([kk[0] for kk in spec_all.keys()]))
    for order in orders_ok:
        wave_t = np.array(list(spec_all[order, 0].values()))
        spec_t = np.array(list(spec_all[order, 1].values()))
        weight_t = np.array(list(spec_all[order, 2].values()))
        weight_t[np.isnan(spec_t)] = 0
        weight_t[spec_t<0] = 0

        wave_tpl_new[order] = wave_t[0]

        if len(spec_t) > 1:
            for nn in range(1, len(spec_t)):
                valid = np.isfinite(spec_t[nn])
                wave_valid = wave_t[nn][valid]
                spec_cubic = CubicSpline(wave_valid, spec_t[nn][valid])(wave_t[0])
                out_of_range = (wave_t[0] < wave_valid[0]) | (wave_t[0] > wave_valid[-1])
                spec_cubic[out_of_range] = np.nan
                spec_t[nn] = spec_cubic
                weight_t[nn] = np.interp(wave_t[0], wave_valid, weight_t[nn][valid])
                weight_t[nn][out_of_range] = np.nan
                weight_t[nn][weight_t[nn]==0] = np.nan

            if kapsig_ctpl:
                spec_mean = np.nanmedian(spec_t, axis=0)
                spec_scatter = np.nanmedian(np.abs(spec_t - spec_mean), axis=0) * 1.4826
                for nn in range(0, len(spec_t)):
                    weight_t[nn][np.abs(spec_t[nn]-spec_mean) > kapsig_ctpl * spec_scatter] = np.nan

            spec_tpl_new[order] = np.nansum(spec_t*weight_t, axis=0) / np.nansum(weight_t, axis=0)
            err_tpl_new[order] = np.nanstd(spec_t, axis=0)

        else:
            spec_tpl_new[order] = spec_t[0]
            err_tpl_new[order] = spec_t[0]*np.nan

        if plot > 0:
            fig2 = plt.figure(2)
            fig2.clf()
            ax = fig2.add_subplot(111)
            ax.plot(wave_tpl_new[order], spec_tpl_new[order] - 1, '-', color='gray', label='combined tpl')
            for n in range(len(spec_t)):
                ax.plot(wave_tpl_new[order], spec_t[n]/np.nanmedian(spec_t[n]), '-', label=os.path.split(obsnames[n])[1])
            ax.set_ylim(-1, 1.6)
            ax.set_xlabel(u'Vacuum wavelength [\u00c5]')
            ax.set_ylabel('flux')
            ax.set_title('order: %s' % order)
            ax.legend(loc='upper right', fontsize='small')
            fig2.tight_layout()
            plt.pause(0.01)
            fig2.savefig('%s_tpl_o%03d.png' % (tag, order), dpi=150)
            if plot == 1:
                input('Press Enter to continue...')

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
