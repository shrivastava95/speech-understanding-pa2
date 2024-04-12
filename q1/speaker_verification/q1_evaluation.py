import torch
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import  brentq
from scipy.interpolate import interp1d
import os

RESULTS_PT_PATH = 'results_q1_inference.pt'
assert os.path.exists(RESULTS_PT_PATH)

results = torch.load(RESULTS_PT_PATH)
PREDICTION_LIST = np.array(results['PREDICTION_LIST'])
MODEL_LIST = results['MODEL_LIST']
GROUND_TRUTH = np.array(results['GROUND_TRUTH'])
EER_LIST = []


def calculate_eer_threshold(fpr, tpr, thresholds):
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer


def thresholded_accuracy(scores, labels, thresh):
    scores, labels = [np.array(arr) for arr in [scores, labels]]
    preds = (scores > thresh).astype(int)
    labels = labels.astype(int)
    accuracy = np.sum(preds == labels) / np.sum(labels == labels)
    return accuracy



for model_name, scores in zip(MODEL_LIST, PREDICTION_LIST):
    labels = GROUND_TRUTH
    fpr, tpr, thresholds = roc_curve(labels, scores)
    threshold, eer_brent = calculate_eer_threshold(fpr, tpr, thresholds)
    eer = thresholded_accuracy(scores, labels, threshold)
    EER_LIST.append(eer_brent)

results["EER_LIST"] = EER_LIST
torch.save(results, 'results_q1_evaluation.pt')







