from __future__ import print_function, division

"""
@ About :
@ Author : Biswajit Satapathy
@ ref. : https://labrosa.ee.columbia.edu/matlab/rastamat/
"""
import warnings
import librosa
import scipy.fftpack as fft
import numpy as np
import spectrum
from scipy import signal

def specgram(x, nfft=256, fs=8000, window=np.array([]), overlap=None):
    """
    input:
        x - input audio_array
        nfft - # fft points [default : 256]
        fs - sampling frequency [default : 8000]
        window - window to FFT analysis [default : hanning(nfft)]
        overlap - overlap for FFT analysis ( overlap = window_length - hop_length ) [default : nfft/2]
    """
    nsamples = x.shape[0]

    if type(window) == int:
        window = np.hanning(window)

    if overlap == None:
        overlap = np.ceil(nfft/2).astype(int)

    if window.shape[0] == 0:
        window = np.hanning(nfft);

    if (nsamples <= nfft):
        raise AssertionError("expected nframes > nfft.")

    # compute window offsets
    win_size = window.shape[0];
    
    if win_size > nfft:
        nfft = win_size
        warnings.warn("fft points adjusted to win_size as win_size > nfft")
    win_step = win_size - overlap


    # build matrix of windowed data slices
    offset = list(range(0,(nsamples-win_size), win_step))
    npad = (nsamples - offset[-1])+1
    if npad > 0:
        x = np.concatenate((x,np.zeros(npad)))

    S = []
    for i in offset:
        S.append((x[i:i+win_size]*window)[np.newaxis,:])
    S=np.concatenate(S)
    S = np.fft.fft(S,nfft)

    # extract the positive frequency components
    ret_n = int(nfft/2);
    if nfft%2==1:
        ret_n = int((nfft+1)/2);

    S = S[:, 0:ret_n];
    f = np.array(range(ret_n))*fs/nfft;
    t = np.array(offset)/fs;
    return (S, f, t)


def powspec(x, fs = 8000,
            winlen_in_sec = 0.025, hoplen_in_sec = 0.010,
            dither = 0
    ):
    """
    pow_spec, e = powspec(x, sr, winlen_in_sec, hoplen_in_sec, dither)
    where    x : audio array
            sr : sampling rate
            winlen_in_sec : window length for audio analysis ()

     compute the powerspectrum and frame energy of the input signal.
     basically outputs a power spectrogram

     each column represents a power spectrum for a given frame
     each row represents a frequency

    """
    # sec2sample
    win_length = int(np.round(winlen_in_sec * fs))
    hop_length = int(np.round(hoplen_in_sec * fs))

    # next power of two of window length is NFFT
    nfft = int(np.power(2, np.ceil(np.log2(winlen_in_sec * fs))))

    specgm,f,t = specgram(x, nfft=nfft, fs=fs, overlap = win_length-hop_length, window=np.hamming(win_length))
    """
    import oct2py
    oc = oct2py.Oct2Py()
    oc.eval('pkg load signal')
    specgm  = oc.specgram(x, nfft, fs, nfft, win_length-hop_length)
    compute short-time FT
    specgm = librosa.stft(
        np.multiply(32768, x),
        n_fft = nfft, hop_length = hop_length,
        win_length = win_length, window='hann',
        center = False
        )
    f, t, specgm = sp.signal.spectrogram(
        x, fs, nfft = nfft, window='hanning',
        noverlap=win_length-hop_length, mode='complex', scaling='density')
    specgm = np.abs(specgram)
    specgm = librosa.stft(
        x,
        n_fft = nfft, hop_length = hop_length,
        win_length = win_length, window='hann',
        center = False
        )
    specgm, f, t = specgram(x, NFFT=nfft, Fs=fs, detrend=None,
        window=None, noverlap=win_length-hop_length, pad_to=None, sides=None, scale_by_freq=None, mode="complex")
    """
    pow_spec = np.power(np.abs(specgm), 2)
    if dither:
        pow_spec = np.add(pow_spec, win_length)

    # Calculate log energy - after windowing
    e = np.log(np.sum(pow_spec, axis = 0))

    return pow_spec, e

def hz2bark(f):
    """
    @About: Converts frequencies Hertz (Hz) to Bark
    It uses;
        Traunmueller-formula    for    f >  200 Hz
        linear mapping          for    f <= 200 Hz

    z_gt_200 = 26.81 .* f ./ (1960 + f) - 0.53;
    z_le_200 = f ./ 102.9;
    z = (f>200) .* z_gt_200 + (f<=200) .* z_le_200;

    @ Author:   Kyrill, Oct. 1996
                Kyrill, March 1997   (linear mapping added)
    """
    # Inverse of Hynek's formula (see bark2hz)
    # z = 6 * log(f/600 + sqrt(1+ ((f/600).^2)));
    # z = 6 * asinh(f/600); # (matlab equivalent)
    z = np.multiply(6.0, np.arcsinh(np.divide(f, 600.0)))
    return z

def bark2hz(z):
    """
    @Author: Converts frequencies Bark to Hertz (Hz)
    It uses;
            Traunmueller-formula    for    z >  2 Bark
            linear mapping          for    z <= 2 Bark

    hz_gt_2 = 1960 .* (z + 0.53) ./ (26.28 - z);
    hz_le_2 = z .* 102.9;
    hz = (z>2) .* hz_gt_2 + (z<=2) .* hz_le_2;

    @Author:    Kyrill, Oct. 1996
                Kyrill, March 1997   (linear mapping added)
    """
    hz = np.multiply(600.0, np.sinh(np.divide(z, 6.0)))
    return hz

def fft2barkmx(
        nfft, sr, nfilts = 0,
        band_width = 1, min_freq = 0.0,
        max_freq = 0.0
        ):
    """
    @About: Generate a matrix of weights to combine FFT bins into Bark bins
    weights = fft2barkmx(nfft, sr, nfilts, width, minfreq, maxfreq)
    where,  nfft : source FFT size
            sr : sampling frequency (Hz)
            nfilts : number of output bands required (else per one bark).
            band_width : a constant width of each band in Bark
            weights : It nfft columns, the second half are all zero.


    Note: Bark spectrum is fft2barkmx(nfft,sr)*abs(fft(xincols,nfft));

    2004-09-05  dpwe@ee.columbia.edu  based on rastamat/audspec.m

    """
    if max_freq == 0:
        max_freq = sr / 2.0

    min_bark = hz2bark(min_freq)
    nyqbark = hz2bark(max_freq) - min_bark

    if nfilts == 0 :
        nfilts = np.add(np.ceil(nyqbark), 1)

    weights = np.zeros((int(nfilts), int(nfft)))
    step_barks = np.divide(nyqbark, np.subtract(nfilts, 1))
    binbarks = hz2bark(np.multiply(np.arange(0, np.add(np.divide(nfft, 2),1)), np.divide(sr, nfft)))

    for i in range (int(nfilts)):
        f_bark_mid = min_bark + np.multiply(i, step_barks)

        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = np.subtract(np.subtract(binbarks, f_bark_mid), 0.5)
        hif = np.add(np.subtract(binbarks, f_bark_mid), 0.5)
        weights[i, 0 : int(nfft / 2) + 1] = np.power(10, np.minimum(0, np.divide(np.minimum(hif, np.multiply(-2.5, lof)), band_width)))

    return weights

def rastafilt(x):
    """
    y = rastafilt(x)
    xrow, xcol = x.shape()

    where,  x : input signal
            xrow : critical bands
            xcol : no of frames
            y : rasta filtered signal

    """
    # rasta filter
    numer = np.arange(-2, 3)
    numer = np.divide(-1.0*numer,sum(np.power(numer,2)))
    denom = np.array([1, -0.94])

    """
    Initialize the state.
    This avoids a big spike at the beginning resulting from the dc offset level in each band.
    """
    zi = signal.lfilter_zi(numer,1)
    y = np.zeros((x.shape))
    for i in range(x.shape[0]):
        # FIR for initial state response compuation
        y1, zi = signal.lfilter(numer, 1, x[i, 0:4], axis = 0, zi = zi * x[i, 0])
        y1 = y1*0

        # IIR
        y2, _ = signal.lfilter(numer, denom, x[i, 4:x.shape[1]], axis = 0, zi = zi)
        y[i, :] = np.append(y1, y2)
    return y


def dolpc(x, modelorder = 8):
    """
    y = dolpc(x,modelorder)
    @About: compute autoregressive model from spectral magnitude samples
    where,  x : input signal
                row_x, col_x = x.shape()
                row_x : critical band
                col_y : nframes

            modelorder : order of model, defaults to 8

            y : lpc coeff.
                row_y, col_y = y.shape()
                row_y :

    """
    nbands, nframes = x.shape
    ncorr = 2 * (nbands - 1)

    # @TO-DO : This need optimisation
    R = np.zeros((ncorr, nframes))
    R[0:nbands, :] = x
    for i in range(nbands - 1):
        R[i + nbands - 1, :] = x[nbands - (i + 1), :]

    # Calculate autocorrelation
    r = fft.ifft(R.T).real.T
    # First half only
    r = r[0:nbands, :]

    y = np.ones((nframes, modelorder + 1))
    e = np.zeros((nframes, 1))

    # Find LPC coeffs by durbin
    if modelorder == 0:
        for i in range(nframes):
            _ , e_tmp, _ = spectrum.LEVINSON(r[:, i], modelorder, allow_singularity = True)
            e[i, 0] = e_tmp
    else:
        for i in range(nframes):
            y_tmp, e_tmp, _ = spectrum.LEVINSON(r[:, i], modelorder, allow_singularity = True)
            y[i, 1:modelorder + 1] = y_tmp
            e[i, 0] = e_tmp

    # Normalize each poly by gain
    y = np.divide(y.T, np.add(np.tile(e.T, (modelorder + 1, 1)), 1e-8))

    return y

def lpc2cep(a, nout = 0):
    """
        cep = lpc2cep(lpcas,nout)
        where,  lpcas = lp coeff.
                nout = number of cepstra to produce, defaults to size(lpcas,1)

    """
    nin, ncol = a.shape
    order = nin - 1

    if nout == 0:
        nout = order + 1

    # First cep is log(Error) from Durbin
    cep = np.zeros((nout, ncol))
    cep[0, :] = -np.log(a[0, :])

    # Renormalize lpc A coeffs
    norm_a = np.divide(a, np.add(np.tile(a[0, :], (nin, 1)), 1e-8))

    for n in range(1, nout):
        sum = 0
        for m in range(1, n):
            sum = np.add(sum, np.multiply(np.multiply((n - m), norm_a[m, :]), cep[(n - m), :]))

        cep[n, :] = -np.add(norm_a[n, :], np.divide(sum, n))

    return cep

def lifter(x, lift = 0.6, invs = False):
    """
    @About : Apply lifter to matrix of cepstra
    y = lifter(x, lift, invs)
    lift = exponent of x i^n liftering or, as a negative integer, the length of HTK-style sin-curve liftering.
    If inverse == True (default False), undo the liftering.

    """
    ncep = x.shape[0]

    if lift == 0:
        y = x
    else:
        if lift < 0:
            warnings.warn('HTK liftering does not support yet; default liftering')
            lift = 0.6
        lift_weights = np.power(np.arange(1, ncep), lift)
        lift_weights = np.append(1, lift_weights)

        if (invs):
            lift_weights = np.divide(1, lift_weights)

        y = np.matmul(np.diag(lift_weights), x)

    return y


def hz2mel(f, htk = False):
    """
    @About : Convert frequencies in Hz to mel 'scale'.

    z = hz2mel(f,htk)
        where,  f : frequency in Hz
                htk : True uses the mel axis defined in the HTKBook otherwise use Slaney's formula

    """

    if htk:
        z = np.multiply(2595, np.log10(np.add(1, np.divide(f, 700))))
    else:
        f_0 = 0.0 # 133.33333;
        f_sp = 200.0 / 3 # 66.66667;
        brkfrq = 1000.0
        # starting mel value for log region
        brkpt = (brkfrq - f_0) / f_sp
        # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
        logstep = np.exp(np.log(6.4) / 27.0)

        f = np.array(f, ndmin = 1)
        z = np.zeros((f.shape[0], ))

        # fill in parts separately
        for i in range(f.shape[0]):
            if f[i] < brkpt:
                z[i] = (f[i] - f_0) / f_sp
            else:
                z[i] = brkpt + (np.log(f[i] / brkfrq) / np.log(logstep))
    return z

def mel2hz(z, htk = False):
    """
    @About : Converts 'mel scale' into Frequency in Hz
    f = mel2hz(z, htk)
        where,  z : frequency in mel scale
                htk : "True" means use the HTK formula else use the formula from Slaney's mfcc
                f : frequency in Hz
    """
    if htk:
        f = np.multiply(700, np.subtract(np.power(10, np.divide(z, 2595)), 1))
    else:
        f_0 = 0.0 # 133.33333;
        f_sp = 200.0/3 # 66.66667;
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp # starting mel value for log region
        logstep = np.exp(np.log(6.4) / 27.0) # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        z = np.array(z, ndmin = 1)
        f = np.zeros((z.shape[0], ))

        # fill in parts separately
        for i in range(z.shape[0]):
            if z[i] < brkpt:
                f[i] = f_0 + f_sp * z[i]
            else:
                f[i] = brkfrq * np.exp(np.log(logstep) * (z[i] - brkpt))
    return f

def fft2melmx(
        nfft, sr=8000, nfilts = 0,
        band_width = 1, min_freq = 0.0,
        max_freq = 0.0, htkmel = False,
        constamp = False
        ):
    """
    @About : Generate a matrix of weights to combine FFT bins into Mel
        bins.
    [weights, binfrqs] = fft2melmx(nfft, sr, nfilts, width, min_freq, max_freq, htkmel, constamp)
        where,  nfft : no of FFT point considered for given sampling rate.
                sr : sampling rate.
                nfilts : number of output bands required (else one per "mel/width")
                width : the constant width of each band relative to standard Mel (default 1).
                minfrq : frequency (in Hz) of the lowest band edge;
                maxfrq : frequency (in Hz) of upper edge; default sr/2.
                htkmel : "True" means use HTK's version of the mel curve, not Slaney's.
                constamp : "True" means make integration windows peak at 1, not sum to 1.

                weights : output model weights. weight has nfft columns, the second half are all zero.
                binfrqs : returns bin center frequencies.

        Note: You can exactly duplicate the mel matrix in Slaney's mfcc.m
        as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    """
    if nfilts == 0 :
        nfilts = np.ceil(hz2mel(max_freq, htkmel) / 2)

    if max_freq == 0:
        max_freq = sr / 2.0

    weights = np.zeros((int(nfilts), int(nfft)))

    # Center freqs of each FFT bin
    fftfrqs = np.multiply(np.divide(np.arange(0,nfft / 2 + 1), nfft), sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz2mel(min_freq, htkmel)
    max_mel = hz2mel(max_freq, htkmel)
    binfrqs = mel2hz(np.add(min_mel, np.multiply(np.arange(0, nfilts + 2),
        (max_mel - min_mel) / (nfilts + 1))), htkmel)

    for i in range (int(nfilts)):
        freq_tmp = binfrqs[np.add(np.arange(0,3), i)]

        # scale by width
        freq_tmp = np.add(freq_tmp[1], np.multiply(band_width, np.subtract(freq_tmp, freq_tmp[1])))
        # lower and upper slopes for all bins
        loslope = np.divide(np.subtract(fftfrqs, freq_tmp[0]), np.subtract(freq_tmp[1], freq_tmp[0]))
        hislope = np.divide(np.subtract(freq_tmp[2], fftfrqs), np.subtract(freq_tmp[2], freq_tmp[1]))
        weights[i, 0 : int(nfft / 2) + 1] = np.maximum(0, np.minimum(loslope, hislope))

    if constamp == False:
        # Slaney-style mel is scaled to be approx constant E per channel
        weights = np.matmul(np.diag(np.divide(2, np.subtract(binfrqs[2 : int(nfilts) + 2],
                                                         binfrqs[0 : int(nfilts)]))), weights)
    return weights, binfrqs

def audspec(p_spectrum,
    fs = 8000, nfilts = 0, fbtype = 'bark',
    min_freq = 0, max_freq = 0, sumpower = 1,
    band_width = 1
    ):
    """
    @About: Performs critical band analysis on power spectrogram (see PLP)

    [aspectrum,weights] = audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth)
        where,  pspectrum : power spectrogram
                sr : sampling frequency
                nfilts : number of output bands required
                fbtype : filterbank type
                minfrq : frequency (in Hz) of the lowest band edge;
                maxfrq : frequency (in Hz) of upper edge; default sr/2.
                sumpower :
                band_width :  the constant width of each band relative to standard Mel (default 1).

                aspectrum : spectrogram aftar band analysis
                weight : output model weights
    """

    if nfilts == 0:
        np.add(np.ceil(hz2bark(fs / 2)), 1)
    if max_freq == 0:
        max_freq = fs / 2
    nframes, nfreqs =  p_spectrum.shape
    # print("nfreq: ", nfreqs, p_spectrum.shape, type(nfreqs))
    nfft = (int(nfreqs) - 1) * 2

    weights = None
    binfrqs = None
    if fbtype == 'bark':
        weights = fft2barkmx(nfft, fs, nfilts, band_width, min_freq, max_freq)
    elif fbtype == 'mel':
        weights,binfrqs = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq)
    elif fbtype == 'htkmel':
        weights,binfrqs = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq, htkmel = True, constamp = True)
    elif fbtype == 'fcmel':
        weights,binfrqs = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq, htkmel = True, constamp = False)
    else:
        raise TypeError("fbtype is not recognised. choose from 'bark', 'mel', 'htkmel', 'fcmel'")

    weights = weights[:, 0 : nfreqs]

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = weights.dot(p_spectrum.T).T
    else:
        aspectrum = np.power(weights.dot(np.sqrt(p_spectrum.T)).T, 2)

    return aspectrum, weights

def postaud(x, fmax, fbtype = 'bark', broaden = 0):
    nbands, nframes = x.shape
    # print("postaud :: ",nbands, nframes)
    nfpts = int(nbands + 2 * broaden)

    if fbtype == 'bark':
        bandcfhz = bark2hz(np.linspace(0, hz2bark(fmax), nfpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax), nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax, htk = True), nfpts), htk = True)

    bandcfhz = bandcfhz[broaden : (nfpts - broaden)]

    # Hynek's magic equal-loudness-curve formula
    fsq = np.power(bandcfhz, 2)
    ftmp = np.add(fsq, 1.6e5)
    eql = np.multiply(np.power(np.divide(fsq, ftmp), 2), np.divide(np.add(fsq, 1.44e6), np.add(fsq, 9.61e6)))

    # weight the critical bands
    z = np.multiply(np.tile(eql, (nframes, 1)).T, x)

    # cube root compress
    z = np.power(z, 0.33)

    # replicate first and last band (because they are unreliable as calculated)
    if broaden:
        y = np.zeros((z.shape[0] + 2, z.shape[1]))
        y[0, :] = z[0, :]
        y[1:nbands + 1, :] = z
        y[nbands + 1, :] = z[z.shape[0] - 1, :]
    else:
        y = np.zeros((z.shape[0], z.shape[1]))
        y[0, :] = z[1, :]
        y[1:nbands - 1, :] = z[1:z.shape[0] - 1, :]
        y[nbands - 1, :] = z[z.shape[0] - 2, :]

    return y.T

def spec2cep(spec, ncep=13, dcttype=2):
    nrow, ncol = spec.shape[0], spec.shape[1]
    dctm = np.zeros((ncep, nrow))

    if dcttype == 2 or dcttype == 3:
        for i in range(ncep):
            dctm[i, :] = np.multiply(np.cos(i*np.arange(1, 2 * nrow, 2)/(2 * nrow)*np.pi), np.sqrt(2 / nrow))

        if dcttype == 2:
            dctm[0, :] = np.divide(dctm[0, :], np.sqrt(2))

    elif dcttype == 4:
        for i in range(ncep):
            dctm[i, :] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(1, nrow + 1)), (nrow + 1)), np.pi)), 2)
            dctm[i, 0] = np.add(dctm[i, 0], 1)
            dctm[i, int(nrow - 1)] = np.multiply(dctm[i, int(nrow - 1)], np.power(-1, i))
        dctm = np.divide(dctm, 2 * (nrow + 1))

    else:
        for i in range(ncep):
            dctm[i, :] = np.divide(np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(0, nrow)), (nrow - 1)), np.pi)), 2), 2 * (nrow - 1))
        dctm[:, 0] = np.divide(dctm[:, 0], 2)
        dctm[:, int(nrow - 1)] = np.divide(dctm[:, int(nrow - 1)], 2)

    # cep = np.matmul(dctm, np.log(np.add(spec, 1e-8)))
    cep = dctm.dot(np.log(np.add(spec, 1e-8)))

    return cep.T, dctm

def lpc2spec(lpcas, nout = 17, FMout = False):

    rows, cols = lpcas.shape
    order = rows - 1

    gg = lpcas[0, :]
    aa = np.divide(lpcas, np.tile(gg, (rows, 1)))

    #    Calculate the actual z-plane polyvals: nout points around unit circle
    tmp_1 = np.array(np.arange(0, nout), ndmin = 2).T
    tmp_1 = np.divide(np.multiply(-1j, np.multiply(tmp_1, np.pi)), (nout - 1))
    tmp_2 = np.array(np.arange(0, order + 1), ndmin = 2)
    zz = np.exp(np.matmul(tmp_1, tmp_2))

    #    Actual polyvals, in power (mag^2)
    features = np.divide(np.power(np.divide(1, np.abs(np.matmul(zz, aa))), 2), np.tile(gg, (nout, 1)))
    F = np.zeros((cols, int(np.ceil(rows / 2))))
    M = F

    if FMout == True:
        for c in range(cols):
            aaa = aa[:, c]
            rr = np.roots(aaa)
            ff_tmp = np.angle(rr)
            ff = np.array(ff_tmp, ndmin = 2).T
            zz = np.exp(np.multiply(1j, np.matmul(ff, np.array(np.arange(0, aaa.shape[0]), ndmin = 2))))
            mags = np.sqrt(np.divide(np.power(np.divide(1, np.abs(np.matmul(zz, np.array(aaa, ndmin = 2).T))), 2), gg[c]))

            ix = np.argsort(ff_tmp)
            dummy = np.sort(ff_tmp)
            tmp_F_list = []
            tmp_M_list = []

            for i in range(ff.shape[0]):
                if dummy[i] > 0:
                    tmp_F_list = np.append(tmp_F_list, dummy[i])
                    tmp_M_list = np.append(tmp_M_list, mags[ix[i]])

            M[c, 0 : tmp_M_list.shape[0]] = tmp_M_list
            F[c, 0 : tmp_F_list.shape[0]] = tmp_F_list

    return features, F, M

def deltas(x, w = 9):
    rows, cols = x.shape
    hlen = np.floor(w / 2)
    win = np.arange(hlen,-(hlen + 1),-1, dtype = 'float32')

    xx = np.append(np.append(np.tile(x[:, 0], (int(hlen), 1)).T, x, axis = 1),
               np.tile(x[:, cols - 1], (int(hlen), 1)).T, axis = 1)

    d = signal.lfilter(win, 1, xx, axis = 1)
    d = d[:, int(2 * hlen) : int(2 * hlen + cols)]
    return d

def cep2spec(cep, nfreq, dcttype = 2):
    ncep, ncol = cep.shape

    dctm  = np.zeros((ncep, nfreq))
    idctm = np.zeros((nfreq, ncep))

    if dcttype == 2 or dcttype == 3:
        for i in range(ncep):
            dctm[i, :] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(1, 2 * nfreq, 2)),
                                                                  (2 * nfreq)), np.pi)), np.sqrt(2 / nfreq))

        if dcttype == 2:
            dctm[0, :] = np.divide(dctm[0, :], np.sqrt(2))
        else:
            dctm[0, :] = np.divide(dctm[0, :], 2)

        idctm = dctm.T

    elif dcttype == 4:
        for i in range(ncep):
            idctm[:, i] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(1, nfreq + 1).T), (nfreq + 1)), np.pi)), 2)

        idctm[:, 0:ncep] = np.divide(idctm[:, 0:ncep], 2)

    else:
        for i in range(ncep):
            idctm[:, i] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(0, nfreq).T), (nfreq - 1)), np.pi)), 2)

        idctm[:, [0, -1]] = np.divide(idctm[:, [0, -1]], 2)

    spec = np.exp(np.matmul(idctm, cep))

    return spec, idctm

def invpostaud(y, fmax, fbtype = 'bark', broaden = 0):

    nbands, nframes = y.shape

    if fbtype == 'bark':
        bandcfhz = bark2hz(np.linspace(0, hz2bark(fmax), nbands))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax), nbands))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(fmax, htk = True), nbands), htk = True)

    bandcfhz = bandcfhz[broaden : (nbands - broaden)]

    fsq = np.power(bandcfhz, 2)
    ftmp = np.add(fsq, 1.6e5)
    eql = np.multiply(np.power(np.divide(fsq, ftmp), 2),
                      np.divide(np.add(fsq, 1.44e6), np.add(fsq, 9.61e6)))

    x = np.power(y, np.divide(1, 0.33))

    if eql[0] == 0:
        eql[0] = eql[1]
        eql[-1] = eql[-2]

    x = np.divide(x[broaden : (nbands - broaden + 1), :], np.add(np.tile(eql.T, (nframes, 1)).T, 1e-8))

    return x, eql

def invpowspec(y, fs, win_time, hoplen_in_sec, excit = []):
    nrow, ncol = y.shape
    r = excit

    winpts = int(np.round(np.multiply(win_time, fs)))
    steppts = int(np.round(np.multiply(hoplen_in_sec, fs)))
    nfft = int(np.power(2, np.ceil(np.divide(np.log(winpts), np.log(2)))))

    # Can't predict librosa stft length...
    tmp = librosa.istft(y, hop_length = steppts, win_length = winpts,
                      window='hann', center = False)
    xlen = len(tmp)
    # xlen = int(np.add(winpts, np.multiply(steppts, np.subtract(ncol, 1))))
    # xlen = int(np.multiply(steppts, np.subtract(ncol, 1)))

    if len(r) == 0:
        r = np.squeeze(np.random.randn(xlen, 1))
    r = r[0:xlen]

    R = librosa.stft(np.divide(r, 32768 * 12), n_fft = nfft, hop_length = steppts,
                     win_length = winpts, window = 'hann', center = False)

    R = np.multiply(R, np.sqrt(y))
    x = librosa.istft(R, hop_length = steppts, win_length = winpts,
                      window = 'hann', center = False)

    return x

def invaudspec(aspectrum, fs = 16000, nfft = 512, fbtype = 'bark',
               min_freq = 0, max_freq = 0, sumpower = True, band_width = 1):

    if max_freq == 0:
        max_freq = fs / 2
    nfilts, nframes = aspectrum.shape

    if fbtype == 'bark':
        weights = fft2barkmx(nfft, fs, nfilts, band_width, min_freq, max_freq)
    elif fbtype == 'mel':
        weights = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq)
    elif fbtype == 'htkmel':
        weights = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq, htkmel = True, constamp = True)
    elif fbtype == 'fcmel':
        weights = fft2melmx(nfft, fs, nfilts, band_width, min_freq, max_freq, htkmel = True, constamp = False)

    weights = weights[:, 0:int(nfft / 2 + 1)]

    ww = np.matmul(weights.T, weights)
    itws = np.divide(weights.T, np.tile(np.maximum(np.divide(np.mean(np.diag(ww)), 100),
                                               np.sum(ww, axis = 0)), (nfilts, 1)).T)
    if sumpower == True:
        spec = np.matmul(itws, aspectrum)
    else:
        spec = np.power(np.matmul(itws, np.sqrt(aspectrum)))

    return spec, weights, itws



def delta_voicebox(CepCoeff, d):
    """
    delta_coeff = mfcc2delta(CepCoeff,d);
    Input:
        CepCoeff: Cepstral Coefficient (Row Represents a feature vector for a frame)
        d       : Lag size for delta feature computation

    Output:
        delta_coeff: Output delta coefficient

    Ref . http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
    """
    vf = np.arange(d,-(d+1),-1)
    vf=vf/sum(vf**2);
    ww=np.ones((d,1));
    NoOfFrame, NoOfCoeff = CepCoeff.shape
    cx = np.concatenate(
        (np.concatenate(CepCoeff[(ww-1).astype(int)]),
        CepCoeff ,
        np.concatenate(CepCoeff[(ww*NoOfFrame-1).astype(int)]))
        )
    cx_r, cx_c = cx.shape
    cx_col = cx.T.reshape(cx_r*cx_c)
    vx = signal.lfilter(vf,1,cx_col.T).reshape((cx_r,cx_c), order='F')
    delta_coeff=vx[2*d::,:]
    return delta_coeff

def sdc(CepCoeff, N=7, D=1, P=3, K=7):
    """
    About: Shifted Delta Coefficient Computation.
    Usage: sdc_coeff = mfcc2sdc(CepCoeff,N,d,P,k)
    input:
        CepCoeff : MFCC Coefficients stored in row-wise
        N: NoOfCepstrum i.e. no of column of CepCoeff
        d: Amount of shift for delta computation
        P: Amount of shift for next frame whose deltas are to be computed.
        K: No of frame whose deltas are to be stacked.

    output:
        sdc_coeff: Shifted delta coefficient of CepCoeff.
        Dimension of the output: NumberOfFrame x N*K

    Ref. W.M. Campbell, J.P.Campbell, D.A. Reynolds, E. Singer, P.A. Torres-Carrasquillo, Support
        vector machines for speaker and language recognition,
        Computer Speech & Language, Volume 20, Issues 2-3,
        Odyssey 2004: The speaker and Language Recognition
        Workshop - Odyssey-04, April-July 2006, Pages 210-229.
    """

    nframe, ncoeff = CepCoeff.shape
    CepCoeff = np.concatenate((CepCoeff,CepCoeff[0:P*(K-1),:]))
    NoOfFrame, NoOfCoeff = CepCoeff.shape
    delta_coeff = delta_voicebox(CepCoeff,D).T
    sdc_coeff = []
    for i in range(K):
        temp=(delta_coeff[:,P*i::])[:,0:nframe]
        sdc_coeff.append(temp)
    sdc_coeff = np.concatenate(sdc_coeff)
    return sdc_coeff
