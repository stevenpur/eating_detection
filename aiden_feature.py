import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import statsmodels.tsa.stattools as stattools

MIN_WINDOW_SEC = 2  # seconds


def aidan_features(xyz, sample_rate, welch_1s=False):
    ''' Extract commonly used HAR time-series features. xyz is a window of shape (N,3) '''

    if np.isnan(xyz).any():
        return {}

    if len(xyz) <= MIN_WINDOW_SEC * sample_rate:
        return {}

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v = np.linalg.norm(xyz, axis=1) - 1

    x, y, z = np.clip(x, -3, 3), np.clip(y, -3, 3), np.clip(z, -3, 3)
    v = np.clip(v, -2, 2)  # clip abnormaly high values
    feats = {}

    # Features for x, y, z, and v
    feats.update(moments_features(x, sample_rate, prefix='x'))
    feats.update(moments_features(y, sample_rate, prefix='y'))
    feats.update(moments_features(z, sample_rate, prefix='z'))
    feats.update(moments_features(v, sample_rate, prefix='v'))

    feats.update(quantile_features(x, sample_rate, prefix='x'))
    feats.update(quantile_features(y, sample_rate, prefix='y'))
    feats.update(quantile_features(z, sample_rate, prefix='z'))
    feats.update(quantile_features(v, sample_rate, prefix='v'))

    feats.update(autocorr_features(x, sample_rate, prefix='x'))
    feats.update(autocorr_features(y, sample_rate, prefix='y'))
    feats.update(autocorr_features(z, sample_rate, prefix='z'))
    feats.update(autocorr_features(v, sample_rate, prefix='v'))

    feats.update(spectral_features(x, sample_rate, prefix='x'))
    feats.update(spectral_features(y, sample_rate, prefix='y'))
    feats.update(spectral_features(z, sample_rate, prefix='z'))
    feats.update(spectral_features(v, sample_rate, prefix='v'))

    if not welch_1s:
        feats.update(fft_features(x, sample_rate, prefix='x'))
        feats.update(fft_features(y, sample_rate, prefix='y'))
        feats.update(fft_features(z, sample_rate, prefix='z'))
        feats.update(fft_features(v, sample_rate, prefix='v'))

    feats.update(peaks_features(x, sample_rate, prefix='x'))
    feats.update(peaks_features(y, sample_rate, prefix='y'))
    feats.update(peaks_features(z, sample_rate, prefix='z'))
    feats.update(peaks_features(v, sample_rate, prefix='v'))

    # Cross-correlation features
    feats.update(cross_corr_features(x, y, 'xy'))
    feats.update(cross_corr_features(y, z, 'yz'))
    feats.update(cross_corr_features(x, z, 'xz'))

    return feats

def moments_features(v, sample_rate=None, prefix=''):
    """ Moments """
    avg = np.mean(v)
    std = np.std(v)
    if std > .01:
        skew = np.nan_to_num(stats.skew(v))
        kurt = np.nan_to_num(stats.kurtosis(v))
    else:
        skew = kurt = 0
    feats = {
        f'{prefix}_avg': avg,
        f'{prefix}_std': std,
        f'{prefix}_skew': skew,
        f'{prefix}_kurt': kurt,
    }
    return feats


def cross_corr_features(x, y, label):
    """ Cross-correlation features between x and y """
    feats = {
        f'corr_{label}': np.nan_to_num(np.corrcoef(x, y)[0, 1]),
    }
    return feats



def moments_features(v, sample_rate=None, prefix=''):
    """ Moments """
    avg = np.mean(v)
    std = np.std(v)
    if std > .01:
        skew = np.nan_to_num(stats.skew(v))
        kurt = np.nan_to_num(stats.kurtosis(v))
    else:
        skew = kurt = 0
    feats = {
        f'{prefix}_avg': avg,
        f'{prefix}_std': std,
        f'{prefix}_skew': skew,
        f'{prefix}_kurt': kurt,
    }
    return feats


def quantile_features(v, sample_rate=None, prefix=''):
    """ Quantiles (min, 25th, med, 75th, max) """
    feats = {}
    feats[f'{prefix}_min'], feats[f'{prefix}_q25'], feats[f'{prefix}_med'], feats[f'{prefix}_q75'], feats[f'{prefix}_max'] = \
        np.quantile(v, (0, .25, .5, .75, 1))
    return feats


def autocorr_features(v, sample_rate, prefix=''):
    """ Autocorrelation features """

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore invalid div warnings
        u = np.nan_to_num(stattools.acf(v, nlags=2 * sample_rate))

    peaks, _ = signal.find_peaks(u, prominence=.1)
    if len(peaks) > 0:
        acf_1st_max_loc = peaks[0]
        acf_1st_max = u[acf_1st_max_loc]
        acf_1st_max_loc /= sample_rate  # in secs
    else:
        acf_1st_max = acf_1st_max_loc = 0.0

    valleys, _ = signal.find_peaks(-u, prominence=.1)
    if len(valleys) > 0:
        acf_1st_min_loc = valleys[0]
        acf_1st_min = u[acf_1st_min_loc]
        acf_1st_min_loc /= sample_rate  # in secs
    else:
        acf_1st_min = acf_1st_min_loc = 0.0

    acf_zeros = np.where(np.diff(np.signbit(u)))
    acf_zeros = len(acf_zeros)

    feats = {
        f'{prefix}_acf_1st_max': acf_1st_max,
        f'{prefix}_acf_1st_max_loc': acf_1st_max_loc,
        f'{prefix}_acf_1st_min': acf_1st_min,
        f'{prefix}_acf_1st_min_loc': acf_1st_min_loc,
        f'{prefix}_acf_zeros': acf_zeros,
    }

    return feats


def spectral_features(v, sample_rate, prefix=''):
    """ Spectral entropy, average power, dominant frequencies """

    feats = {}

    freqs, powers = signal.periodogram(v, fs=sample_rate, detrend='constant', scaling='density')
    powers /= (len(v) / sample_rate)    # unit/sec

    feats[f'{prefix}_pentropy'] = stats.entropy(powers[powers > 0])
    feats[f'{prefix}_power'] = np.sum(powers)

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]

    TOPN = 3
    feats.update({f"{prefix}_f{i + 1}": 0 for i in range(TOPN)})
    feats.update({f"{prefix}_p{i + 1}": 0 for i in range(TOPN)})
    for i, j in enumerate(peak_ranks[:TOPN]):
        feats[f"{prefix}_f{i + 1}"] = peak_freqs[j]
        feats[f"{prefix}_p{i + 1}"] = peak_powers[j]

    return feats


def fft_features(v, sample_rate, nfreqs=5, prefix=''):
    """ Power of frequencies 0Hz, 1Hz, 2Hz, ... using Welch's method """

    _, powers = signal.welch(
        v, fs=sample_rate,
        nperseg=sample_rate,
        noverlap=sample_rate // 2,
        detrend='constant',
        scaling='density',
        average='median'
    )

    feats = {f"{prefix}_fft{i}": powers[i] for i in range(nfreqs + 1)}

    return feats


def peaks_features(v, sample_rate, prefix=''):
    """ Features of the signal peaks """

    feats = {}
    u = butterfilt(v, 5, fs=sample_rate)  # lowpass 5Hz
    peaks, peak_props = signal.find_peaks(u, distance=0.2 * sample_rate, prominence=0.25)
    feats[f'{prefix}_npeaks'] = len(peaks) / (len(v) / sample_rate)  # peaks/sec
    if len(peak_props['prominences']) > 0:
        feats[f'{prefix}_peaks_avg_promin'] = np.mean(peak_props['prominences'])
        feats[f'{prefix}_peaks_min_promin'] = np.min(peak_props['prominences'])
        feats[f'{prefix}_peaks_max_promin'] = np.max(peak_props['prominences'])
    else:
        feats[f'{prefix}_peaks_avg_promin'] = feats[f'{prefix}_peaks_min_promin'] = feats[f'{prefix}_peaks_max_promin'] = 0

    return feats


def butterfilt(x, cutoffs, fs, order=4, axis=0):
    """ Butterworth filter """
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            btype = 'bandpass'
            Wn = (hicut / nyq, lowcut / nyq)
        else:
            btype = 'low'
            Wn = lowcut / nyq
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y


def get_feature_names():
    """ Hacky way to get the list of feature names """

    feats = aidan_features(np.zeros((500, 3)), 100)
    return list(feats.keys())


if __name__ == "__main__":
    data = np.random.rand(1000, 3)

    sample_rate = 100
    # Extract features
    features = aidan_features(data, sample_rate)

    # Print or use the extracted features as needed
    print(features)