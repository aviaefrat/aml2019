import os
import pickle

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import RESULTS_DIR, FEATURE_TYPES, LANGUAGES
from main import _recursive_defaultdict

def print_accuracies(results_filename='results.pkl', feature_types_=FEATURE_TYPES):
    with open(os.path.join(RESULTS_DIR, results_filename), 'rb') as f:
        results = pickle.load(f)
        errors = {}
        for feature_type in feature_types_:
            rows = []
            for preprocessing, feature_types in results.items():
                rows.append(pd.Series([1 - feature_types[feature_type]['train_error'],
                                       1 - feature_types[feature_type]['test_error']],
                                      name=preprocessing))
            errors[feature_type] = pd.DataFrame(rows)
            errors[feature_type].columns = ['train_acc', 'test_acc']
            errors[feature_type].sort_index(axis=0, inplace=True)
    print(errors)


def plot_confusion_matrix(feature_type, preprocessing_type, results_filename='results.pkl', normalize=True):
    with open(os.path.join(RESULTS_DIR, results_filename), 'rb') as f:
        results = pickle.load(f)
        y_true = results[preprocessing_type]['y_test'].map({v: k for k, v in LANGUAGES.items()})
        y_pred = results[preprocessing_type][feature_type]['test_pred'].map({v: k for k, v in LANGUAGES.items()})
    # title = f"features: {feature_type}   ;   preprocessing: {preprocessing_type}"
    _plot_confusion_matrix(y_true, y_pred, LANGUAGES.keys(), normalize=normalize)


def _plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


print_accuracies('results_ngrams_cv5_spaces_dashes_aposts.pkl', feature_types_=['ngrams'])


plot_confusion_matrix('ngrams', 'rt-handle-url-red_rep', results_filename='results_ngrams_cv5_spaces_dashes_aposts.pkl')
plot_confusion_matrix('ngrams', 'rt-handle-url-red_rep', results_filename='results_ngrams_cv5_spaces_dashes_aposts.pkl',normalize=False)
plt.show()

