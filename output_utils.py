import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from PIL import Image
import matplotlib.pyplot as plt
import re
import bisect
from sklearn import metrics

def attach_mode_smooth(model_metrics, window_size):
    '''Record the window size of mode smoothing'''
    model_metrics['is_smoothed'] = True
    model_metrics['smoothing'] = {
        "method": "mode",
        "window_size": window_size
    }
    return model_metrics

def attach_hmm_smooth(model_metrics, hmm_params, class_labels):
    '''Record the parameters of HMM smoothing and label them with class names in Pandas DataFrames'''
    # check if the pior, emission and transition exist and are numpy arrays
    if not all([isinstance(hmm_params[key], np.ndarray) for key in ['prior', 'emission', 'transition']]):
        raise ValueError("hmm_params must contain numpy arrays for 'prior', 'emission', and 'transition'")
    # Convert numpy arrays to Pandas DataFrames with appropriate labels
    hmm_params['prior'] = pd.DataFrame(hmm_params['prior'], index=class_labels, columns=['prior'])
    hmm_params['emission'] = pd.DataFrame(hmm_params['emission'], index=class_labels, columns=class_labels)
    hmm_params['transition'] = pd.DataFrame(hmm_params['transition'], index=class_labels, columns=class_labels)

    model_metrics['is_smoothed'] = True
    model_metrics['smoothing'] = {
        "method": "HMM",
        "hmm_params": hmm_params
    }
    return model_metrics
    
    
def record_performance(estimator, y_test, y_test_predicted, labels):
    """Record performance metrics for a given estimator and test set"""
    model_metrics = {
        'estimator': estimator,
        'is_smoothed': False,
        'parameters': estimator.get_params(),
        'metrics': None,  # Placeholder for class-wise metrics
        'confusion_matrix': {},  # Separate storage for confusion matrices
    }
    # Compute the overall confusion matrix
    cm = metrics.confusion_matrix(y_test, y_test_predicted)
    model_metrics['confusion_matrix']['overall'] = pd.DataFrame(cm, index=labels, columns=labels)  
    sum_rows = cm.sum(axis=1)
    sum_columns = cm.sum(axis=0)
    total = cm.sum() 
    # Initialize class-wise metrics
    print(labels, flush=True)
    for label in labels:
        print("yo", flush=True)
        print(label, type(label), flush=True)
    class_metrics = {label: {} for label in labels}
    specificity_sum = 0
    recall_sum = 0
    # Compute and store confusion matrices for each class
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = sum_columns[i] - tp
        fn = sum_rows[i] - tp
        tn = total - tp - fp - fn
        class_cm = pd.DataFrame({
            label: [tp, fn],
            'Not ' + label: [fp, tn]
        }, index=[label, 'Not ' + label])
        model_metrics['confusion_matrix'][label] = class_cm
        # Compute and store metrics for each class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        balanced_accuracy = (specificity + recall) / 2
        # Record class-wise metrics
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'specificity': specificity,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
        }
        # Update overall metrics
        specificity_sum += specificity * sum_rows[i]
        recall_sum += recall * sum_rows[i]
    # Compute overall metrics
    overall_specificity = specificity_sum / total
    overall_recall = recall_sum / total
    overall_balanced_accuracy = (overall_specificity + overall_recall) / 2
    # Record overall metrics
    class_metrics['overall'] = {
        'precision': metrics.precision_score(y_test, y_test_predicted, average='weighted'),
        'recall': overall_recall,
        'f1-score': metrics.f1_score(y_test, y_test_predicted, average='weighted'),
        'specificity': overall_specificity,
        'accuracy': metrics.accuracy_score(y_test, y_test_predicted),
        'balanced_accuracy': overall_balanced_accuracy,
    }
    # Convert class-wise metrics to DataFrame
    model_metrics['metrics'] = pd.DataFrame(class_metrics).transpose()
    model_metrics['kappa'] = metrics.cohen_kappa_score(y_test, y_test_predicted)
    return model_metrics


def dict_to_md(model_metrics, level = 1):
    """convert a dictionary of metrics to a markdown string"""
    # iterate through the dictionary
    md = ""
    for key, value in model_metrics.items():
        if isinstance(value, dict):
            md += f"{'#' * level} {key}\n\n"
            md += dict_to_md(value, level + 1)
        elif isinstance(value, pd.DataFrame):
            md += f"{'####'} {key}\n"
            md += value.to_markdown() + '\n'
        # now checkk if it is a numpy array    
        elif isinstance(value, np.ndarray):
            md += f"{'####*'} {key}\n"
            md += pd.DataFrame(value).to_markdown() + '\n'
        else:
            md += f"**{key}**:"
            md += f"{value}\n"
    return md

def extract_time_from_img(img_file):
    img_split = img_file.replace(".JPG", "").split("_")
    # get rid of alphabetical characters
    time = re.sub(r'[a-zA-Z]', '', img_split[-1])
    time = int(time)
    return time

def fetch_camera_jpg(ind, time):
    """get the capture24 image file for a given patient and time"""
    env_path = "../"
    load_dotenv(dotenv_path = env_path + ".env")
    img_path = os.getenv("CAPTURE24_PATH") + "camera/" + ind + "/"
    # get time format in HourMinuteSecond
    if isinstance(time, np.datetime64):
        time_str = str(time)
        time = time_str[11:13] + time_str[14:16] + time_str[17:19]
        date = time_str[0:4] + time_str[5:7] + time_str[8:10]
    else:
        raise ValueError("time must be a np.datetime64 object")
    # grep the image file containting the date
    img_files = [f for f in os.listdir(img_path) if date in f]
    # extract the time from the image file
    img_files_time =[extract_time_from_img(f) for f in img_files]
    img_files_time, img_files = zip(*sorted(zip(img_files_time, img_files)))
    # find the image file that falls in the time range of interest
    img_index = bisect.bisect_left(img_files_time, int(time))
    print(img_index, flush = True)
    img_path = [img_path + img_files[img_index]]
    return img_path

def view_mistake(y_test_predicted, y_test, t_test, pid_test, time_slot_seleciton = "earliest", window_size = 30):
    """view the capture24 image data for the cases where the model made a mistake,
    also plot the trace of the accelerometer data"""
    # get the indices of the mistakes
    mistakes = np.where(y_test_predicted != y_test)
    if (time_slot_seleciton == "earliest"):
        mistake = mistakes[0][0]
    elif (time_slot_seleciton == "latest"):
        mistake = mistakes[0][-1]
    elif (time_slot_seleciton == "random"):
        mistake = np.random.choice(mistakes[0])
    else:
        raise ValueError("time_slot_seleciton must be 'earliest', 'latest', or 'random'")
    # convert the mistake intex to time
    mistake_time = t_test[mistake]
    mistake_pid = pid_test[mistake]
    # load the image
    img_file = fetch_camera_jpg(mistake_pid, mistake_time)
    print(f"mistake time: {mistake_time}")
    print(f"mistake pid: {mistake_pid}")
    print(f"image file: {img_file}")
