import numpy as np
from hparams import hparams

epsilon = 0.0000000001

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy import interp

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def Cmatrix(rater_a, rater_b, min_rating=0, max_rating=1):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat



def histogram(ratings, min_rating=0, max_rating=1):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def accuracy_metrics(labels, pred_logits, plot_auc=None, plot_path='temp', best_thresh=None):
    if hparams.cuda:
        labels = labels.cpu()
        pred_logits = pred_logits.cpu()

    labels = labels.numpy()
    single_labels = labels
    labels = indices_to_one_hot(labels, hparams.num_classes)
    pred_logits = pred_logits.detach().numpy()

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresh = [[]]*hparams.num_classes
    for i in range(hparams.num_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(labels[:, i], pred_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), pred_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(hparams.num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(hparams.num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= hparams.num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    if plot_auc:

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro (auc- {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro (auc- {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['blue', 'green', 'red', 'cyan', 'yellow', 'magenta', 'black'])
        for i, color in zip(range(hparams.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label='{0} (auc- {1:0.2f})'
                     ''.format(hparams.id_to_class[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC AUC curve for {} dataset'.format(plot_auc))
        plt.legend(loc="lower right")
        print(plot_path)
        plt.savefig(plot_path+'_roc_curve.png')

    pred_labels = np.argmax(pred_logits, axis=1)
    one_hot_pred_labels = indices_to_one_hot(pred_labels, hparams.num_classes)
    acc = {'avg': accuracy_score(single_labels, pred_labels)}
    acc_sum = 0
    for idx in range(hparams.num_classes):
        mask = (one_hot_pred_labels[:,idx] == 1) | (labels[:,idx] == 1)
        acc[idx] = accuracy_score(labels[mask,idx], one_hot_pred_labels[mask,idx])
        acc_sum += acc[idx]
    acc['man_avg'] = acc_sum/10

    f1 = f1_score(single_labels, pred_labels, average=None)
    f1 = {idx: f1[idx] for idx in range(hparams.num_classes)}
    f1['micro'] = f1_score(single_labels, pred_labels, average='micro')
    f1['macro'] = f1_score(single_labels, pred_labels, average='macro')

    conf_mat = confusion_matrix(single_labels, pred_labels, labels=range(hparams.num_classes))

    return roc_auc, f1, acc, conf_mat

#
# import torch
# import random
#
# torch.manual_seed(0)
#
# batch_sz = 320
# num_classes = 1
#
# pred_logits = torch.randn((batch_sz, 10))
# labels = torch.randint(0,10,(batch_sz,num_classes))
#
# auc, f1, acc, conf_mat = accuracy_metrics(labels, pred_logits, plot_auc='temp')
#
# print(auc)
# print(f1)
# print(acc)
# print(conf_mat)
