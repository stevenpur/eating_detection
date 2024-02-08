import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, confusion_matrix



def load_data(datafile):
    data = pd.read_csv(
        datafile,
        dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
    )
    data['time'] = pd.to_datetime(data['time'], format = "ISO8601")
    data.set_index('time', inplace=True)
    return data

def make_windows(data, winsec=30, sample_rate=100, dropna=True, verbose=False):

    X, Y, T = [], [], []

    for t, w in tqdm(data.resample(f"{winsec}s", origin='start'), disable=not verbose):

        if len(w) < 1:
            continue

        t = t.to_numpy()

        x = w[['x', 'y', 'z']].to_numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            y = w['annotation'].mode(dropna=False).iloc[0]

        if dropna and pd.isna(y):  # skip if annotation is NA
            continue

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        X.append(x)
        Y.append(y)
        T.append(t)

    X = np.stack(X)
    Y = np.stack(Y)
    T = np.stack(T)

    return X, Y, T


def is_good_window(x, sample_rate, winsec):
    ''' Check there are no NaNs and len is good '''

    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def plot_compare(t, y_true, y_pred, trace=None, min_trace=0, max_trace=1):

    if trace is not None:  # normalize
        if isinstance(trace, (pd.DataFrame, pd.Series)):
            trace = trace.to_numpy()
        trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))

    # uniform resampling
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'trace': trace}, index=t).asfreq('30s')
    y_true, y_pred = data[['y_true', 'y_pred']].to_numpy().T.astype('str')
    trace = data['trace'].to_numpy()
    t = data.index.to_numpy()

    LABEL_COLOR = {
        "sleep": "tab:purple",
        "sit-stand": "tab:red",
        "vehicle": "tab:brown",
        "mixed": "tab:orange",
        "walking": "tab:green",
        "bicycling": "tab:olive",
        "eating": "tab:pink",
        "maybe-eating": "tab:blue",
        # set nan to black color
        "nan": "tab:gray",
    }

    def ax_plot(ax, t, y, ylabel=None):
        labels = list(LABEL_COLOR.keys())
        colors = list(LABEL_COLOR.values())

        y = max_trace * (y[:, None] == labels)

        ax.stackplot(t, y.T, labels=labels, colors=colors)

        ax.set_ylabel(ylabel)
        ax.set_ylim((min_trace, max_trace))
        ax.set_yticks([])

        ax.xaxis.grid(True, which='major', color='k', alpha=0.5)
        ax.xaxis.grid(True, which='minor', color='k', alpha=0.25)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(0,24,4)))
        ax.xaxis.set_minor_locator(mpl.dates.HourLocator())

        ax.tick_params(labelbottom=False, labeltop=True, labelsize=8)
        ax.set_facecolor('#d3d3d3')

        ax.plot(t, trace, c='k')

    fig, axs = plt.subplots(nrows=3, figsize=(10, 3))
    ax_plot(axs[0], t, y_true, ylabel='true')
    ax_plot(axs[1], t, y_pred, ylabel='pred')
    axs[1].set_xticklabels([])  # hide ticks for second row

    # legends
    axs[-1].axis('off')
    legend_patches = [mpatches.Patch(facecolor=color, label=label)
                      for label, color in LABEL_COLOR.items()]
    axs[-1].legend(handles=legend_patches,
                   bbox_to_anchor=(0., 0., 1., 1.),
                   ncol=3,
                   loc='center',
                   mode="best",
                   borderaxespad=0,
                   framealpha=0.6,
                   frameon=True,
                   fancybox=True)

    return fig, axs


def train_hmm(Y_prob, Y_true, labels, uniform_prior=True):
    ''' https://en.wikipedia.org/wiki/Hidden_Markov_model '''

    if uniform_prior:
        # All labels with equal probability
        prior = np.ones(len(labels)) / len(labels)
    else:
        # Label probability equals observed rate
        print("hi")
        prior = np.mean(Y_true.reshape(-1,1)==labels, axis=0)

    emission = np.vstack(
        [np.mean(Y_prob[Y_true==label], axis=0) for label in labels]
    )
    transition = np.vstack(
        [np.mean(Y_true[1:][(Y_true==label)[:-1]].reshape(-1,1)==labels, axis=0)
            for label in labels]
    )

    params = {'prior':prior, 'emission':emission, 'transition':transition, 'labels':labels}

    return params


def viterbi(Y_obs, hmm_params):
    ''' https://en.wikipedia.org/wiki/Viterbi_algorithm '''

    def log(x):
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y_obs)
    nlabels = len(labels)

    Y_obs = np.where(Y_obs.reshape(-1,1)==labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0,:] = log(prior) + log(emission[:, Y_obs[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j,i] = np.max(
                log(emission[i, Y_obs[j]]) + \
                log(transition[:, i]) + \
                probs[j-1,:])  # probs already in log scale
    viterbi_path = np.zeros_like(Y_obs)
    viterbi_path[-1] = np.argmax(probs[-1,:])
    for j in reversed(range(nobs-1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j+1]]) + \
            probs[j,:])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path


def ewm(x, alpha=.05):
    """ Exponentially weighted mean """
    n = len(x)
    weights = np.asarray([(1 - alpha)**i for i in range(n)])[::-1]
    weights[weights < 1e-3] = 0  # ignore very small weights
    return (x * weights).sum() / weights.sum()

def get_class_index(y_true, class_label):
    classes = np.unique(y_true)
    return np.where(classes == class_label)[0][0]

def specificity_multiclass(y_true, y_pred, class_index):
    """
    Compute the specificity for a given class index in a multi-class setting.
    """
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    total_negative = cm.sum() - cm[class_index].sum()
    true_negative = total_negative - cm[:, class_index].sum() + cm[class_index, class_index]
    specificity = true_negative / total_negative if total_negative else 0
    return specificity


def class_specific_ba(y_true, y_pred, class_label):
    """
    Compute the balanced accuracy (average of recall and specificity) for a given class in a multi-class setting.
    """
    recall = recall_score(y_true, y_pred, labels=[class_label], average=None)
    class_index = get_class_index(y_true, class_label)
    spec = specificity_multiclass(y_true, y_pred, class_index)
    return (recall + spec) / 2

def make_ba_scorer_for_class(class_label):
    """
    Create a balanced accuracy scorer for a given class in a multi-class setting.
    """
    def scorer(y_true, y_pred):
        return class_specific_ba(y_true, y_pred, class_label)
    return make_scorer(scorer)