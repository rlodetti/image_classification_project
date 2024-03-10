import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, metrics, models, regularizers
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Set the random seeds for reproducibility across libraries
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define metrics globally to avoid redundancy
METRICS = [
    metrics.AUC(name='auc'),
    metrics.Recall(name='recall'),
    metrics.BinaryAccuracy(name='accuracy')
]

def trim_elements(input_dict):
    """Trims the last 10 elements from each list in a dictionary.

    Args:
        input_dict (dict): A dictionary containing lists as values.

    Returns:
        dict: A dictionary with trimmed lists.
    """
    return {key: value[:-10] for key, value in input_dict.items()}

def load_viz(results, num_epochs):
    """Visualizes the training and validation metrics and loss over epochs.

    Args:
        results (dict): The history object returned from a model's fit method.
        num_epochs (int): The number of epochs used in training.
    """
    epochs_range = range(num_epochs)
    plt.figure(figsize=(12, 6))

    # Metrics plot
    for metric in ['recall', 'loss']:
        plt.subplot(1, 2, ['recall', 'loss'].index(metric) + 1)
        plt.plot(epochs_range, results[metric], label=f'Training {metric.capitalize()}')
        plt.plot(epochs_range, results[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.legend(loc='upper right' if metric == 'loss' else 'lower right')
        plt.title(f'Training and Validation {metric.capitalize()}')
    plt.show()

def model_loader(model_path, results_path):
    """Loads a model and its training results.

    Args:
        model_path (str): Path to the saved model.
        results_path (str): Path to the pickled training results.

    Returns:
        tuple: The training results dictionary, loaded model, and number of epochs.
    """
    with open(results_path, 'rb') as file_handle:
        results = pickle.load(file_handle)
    
    model = load_model(model_path)
    num_epochs = len(results['loss'])
    
    return results, model, num_epochs

def evaluate_and_print_results(model_path, results_path, train_ds, val_ds):
    """Loads a model, visualizes its performance, and prints training and validation scores.

    Args:
        model_path (str): Path to the saved model.
        results_path (str): Path to the pickled training results.
        train_ds: Training dataset.
        val_ds: Validation dataset.
    """
    results, model, num_epochs = load_model_and_results(model_path, results_path)
    load_viz(results, num_epochs)
    
    # Evaluation
    train_scores = model.evaluate(train_ds, verbose=0)
    val_scores = model.evaluate(val_ds, verbose=0)
    
    # Print results
    metrics_names = [metric.name for metric in METRICS]
    performance_df = pd.DataFrame([train_scores, val_scores], 
                                  index=['Train', 'Validation'], 
                                  columns=metrics_names)
    print(performance_df)
    print('------------------------------\n')

def summary_viz(model_name, val_ds, early_stopping=False):
    """Generates a summary visualization for a model.

    Args:
        model_name (str): The name of the model for loading and saving.
        val_ds: The validation dataset for evaluation.
        early_stopping (bool): Indicates if early stopping was used.

    Returns:
        The loaded Keras model.
    """
    base_dir = Path('saved_models')
    model_path = base_dir / f'{model_name}.keras'
    results_path = base_dir / f'{model_name}.pkl'
    
    results, model, num_epochs = model_loader(str(model_path), str(results_path))
    if early_stopping:
        results = trim_elements(results)
        num_epochs -= 10
    load_viz(results, num_epochs)
    
    val_scores = model.evaluate(val_ds, verbose=0)
    loss = round(val_scores[0],4)
    recall = round(val_scores[2],4)
    auc = round(val_scores[1],4)
    scores = [loss, recall, auc]
    metrics_summary = pd.DataFrame([scores], 
                                   columns=['Loss', 'Recall', 'AUC'], 
                                   index=[model_name])
    display(metrics_summary)
    return model

def compare_models_df(file_names, model_names, val_ds):
    """Compares multiple models by evaluating them on a validation dataset.

    Args:
        file_names (list): List of file names for the models to compare.
        model_names (list): List of model names for display.
        val_ds: Validation dataset for evaluation.
    """
    base_dir = Path('saved_models')
    scores = []
    
    for file_name in file_names:
        model_path = base_dir / f'{file_name}.keras'
        results_path = base_dir / f'{file_name}.pkl'
        
        results, model, num_epochs = model_loader(model_path, results_path)
        
        val_scores = model.evaluate(val_ds, verbose=0)
        loss = round(val_scores[0],4)
        recall = round(val_scores[2],4)
        AUC = round(val_scores[1],4)
        scores.append([loss, recall, AUC])
    
    summary_df = pd.DataFrame(scores, columns=['Loss', 'AUC', 'Recall'], index=model_names)
    display(summary_df)

# Creating a function to predict and plot a confusion matrix
def con_mat(model, dataset):
    images_list = []
    labels_list = []
    
    # Iterate over the test dataset
    for images, labels in dataset:
        # Convert the current batch to numpy and append it to the lists
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())
    
    # Concatenate all batches to create single arrays
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
                                 
    y_preds = (model.predict(images) >= 0.5).astype(int)
    y_test = labels

    cnf_matrix = confusion_matrix(y_test,y_preds,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.show()
