from __future__ import print_function, division
"""
@ About :
@ Author : Biswajit Satapathy
@ ref. : https://labrosa.ee.columbia.edu/matlab/rastamat/
"""
from . import utils as rasta_utils
from scipy import signal
import numpy as np

def rastaplp(x, fs = 8000, window_time = 0.025, hop_time = 0.010, dorasta = True, modelorder = 8):
    # TODO : Need to be fixed compuation process
    # first compute power spectrum
    pspectrum, logE = rasta_utils.powspec(x, fs = fs, winlen_in_sec = window_time, hoplen_in_sec = hop_time)
    # next group to critical bands
    aspectrum, _ = rasta_utils.audspec(pspectrum, fs = fs, min_freq = 0, max_freq = fs/2)
    nbands = aspectrum.shape[0]

    if dorasta:
        # put in log domain
        nl_aspectrum = np.log(aspectrum)
        # next do rasta filtering
        ras_nl_aspectrum = rasta_utils.rastafilt(nl_aspectrum)
        # do inverse log
        aspectrum = np.exp(ras_nl_aspectrum)

    postspectrum = rasta_utils.postaud(aspectrum.T, fmax = fs/2)


    lpcas = rasta_utils.dolpc(postspectrum.T, modelorder)
    cepstra = rasta_utils.lpc2cep(lpcas, nout = modelorder+1)

    if modelorder > 0:
        lpcas = rasta_utils.dolpc(postspectrum.T, modelorder)
        cepstra = rasta_utils.lpc2cep(lpcas, modelorder + 1)
        spectra,F,M = rasta_utils.lpc2spec(lpcas, nbands)
    else:
        cepstra = rasta_utils.spec2cep(spectra)

    cepstra = rasta_utils.lifter(cepstra, 0.6).T

    return cepstra

def melfcc(x, fs = 16000, min_freq = 50, max_freq = 6500, n_mfcc = 13, n_bands = 40, lifterexp = 0.6,
          fbtype = 'fcmel', dcttype = 1, usecmp = False, window_time = 0.025, hop_time = 0.010,
          preemph = 0.97, dither = 0, sumpower = 1, band_width = 1, modelorder = 0,
           broaden = 0, useenergy = False):

    if preemph != 0:
        b = [1, -preemph]
        a = 1
        x = signal.lfilter(b, a, x)

    pspectrum, logE = rasta_utils.powspec(x, fs = fs, winlen_in_sec = window_time, hoplen_in_sec = hop_time, dither = dither)

    aspectrum, _ = rasta_utils.audspec(pspectrum, fs = fs, nfilts = n_bands, fbtype = fbtype,
                        min_freq = min_freq, max_freq = max_freq)

    if usecmp:
        aspectrum = rasta_utils.postaud(aspectrum.T, fmax = max_freq, fbtype = fbtype)

    if modelorder > 0:
        lpcas = rasta_utils.dolpc(aspectrum.T, modelorder)
        cepstra = rasta_utils.lpc2cep(lpcas, nout = n_mfcc)
    else:
        cepstra, dctm = rasta_utils.spec2cep(aspectrum.T, ncep = n_mfcc, dcttype = dcttype)

    cepstra = rasta_utils.lifter(cepstra.T, lift = lifterexp)

    if useenergy == True:
        cepstra[0, :] = logE

    return cepstra.T


def invmelfcc(cep, fs, win_time = 0.040, hop_time = 0.020, lifterexp = 0.6, sumpower = True,
             preemph = 0.97, max_freq = 6500, min_freq = 50, n_bands = 40, band_width = 1,
             dcttype = 2, fbtype = 'mel', usecmp = False, modelorder = 0, broaden = 0, excitation = []):

    winpts = int(np.round(np.multiply(win_time, fs)))
    nfft = int(np.power(2, np.ceil(np.divide(np.log(winpts), np.log(2)))))

    cep = rasta_utils.lifter(cep, lift = lifterexp, invs = True)

    pspc, _ = rasta_utils.cep2spec(cep, nfreq = int(n_bands + 2 * broaden), dcttype = dcttype)

    if usecmp == True:
        aspc, _ = rasta_utils.invpostaud(pspc, fmax = max_freq, fbtype = fbtype, broaden = broaden)
    else:
        aspc = pspc

    spec, _, _ = rasta_utils.invaudspec(aspc, fs = fs, nfft = nfft, fbtype = fbtype, min_freq = min_freq,
                            max_freq = max_freq, sumpower = sumpower, band_width = band_width)

    x = rasta_utils.invpowspec(spec, fs, win_time = win_time, hop_time = hop_time, excit = excitation)

    if preemph != 0:
        b = [1, -preemph]
        a = 1
        x = signal.lfilter(b, a, x)

    return x, aspc, spec, pspc
