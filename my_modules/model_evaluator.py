import time
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import models, layers, metrics, callbacks, regularizers
import pickle
from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np
import random

# Set the random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define the metrics outside the function to avoid redundancy
METRICS = [
    metrics.AUC(name='auc'),
    metrics.Recall(name='recall'),
    metrics.BinaryAccuracy(name='accuracy')]

def visualize_training_results(results, num_epochs):
    """
    Visualizes training results including accuracy/recall/auc and loss over epochs.

    Parameters:
    - results: The History object returned from the fit method of the model.
    - num_epochs: Number of epochs the model was trained for.
    """
    metric_keys = list(results.history.keys())
    metric = metric_keys[3]  # Assuming the first metric after loss and val_loss
    train_metric = results.history[metric]
    val_metric = results.history['val_' + metric]

    epochs_range = range(num_epochs)

    plt.figure(figsize=(10, 5))

    # Plotting metric
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_metric, label=f'Training {metric}')
    plt.plot(epochs_range, val_metric, label=f'Validation {metric}')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation {metric}')

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, results.history['loss'], label='Training Loss')
    plt.plot(epochs_range, results.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def save_model(model, results, file_name, file_path):
    full_path = file_path+file_name
    
    model.save(full_path+'.keras')

    with open(full_path+'.pkl', 'wb') as f:
        pickle.dump(results.history, f)

def modeler(model, file_name, file_path, train=None, val=None, metrics=METRICS, 
            optimizer='adam', num_epochs=10, early_stopping=None):
    """
    Compiles, trains, and evaluates the model, and visualizes training results.

    Parameters:
    - model: The Keras model to be trained.
    - metrics: List of metrics to be evaluated by the model during training and testing.
    - optimizer: String (name of optimizer) or optimizer instance to use.
    - num_epochs: Number of epochs to train the model.
    - early_stopping: EarlyStopping callback to stop training early if no improvement.
    - train: Training dataset.
    - val: Validation dataset.
    """
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    start_time = time.time()
    callbacks_list = [early_stopping] if early_stopping else []
    results = model.fit(train, epochs=num_epochs, validation_data=val, verbose=0,
                        callbacks=callbacks_list)
    end_time = time.time()

    save_model(model, results, file_name, file_path)
    
    print(f"Training time: {end_time - start_time} seconds\n")
    visualize_training_results(results, num_epochs=int(len(results.history['loss'])))

    # Evaluate model
    train_scores = model.evaluate(train, verbose=0)
    val_scores = model.evaluate(val, verbose=0)
    
    # Display performance difference
    metrics_names = ['loss'] + [metric.name for metric in metrics]
    diff_scores = [val - train for train, val in zip(train_scores, val_scores)]
    performance_df = DataFrame([train_scores, val_scores, diff_scores], 
                               index=['Train', 'Val', 'Diff'], columns=metrics_names)
    display(performance_df)
    print('------------------------------\n')

    return results, model


def load_viz(results, num_epochs):
    """
    Visualizes training results including accuracy/recall/auc and loss over epochs.

    Parameters:
    - results: The History object returned from the fit method of the model.
    - num_epochs: Number of epochs the model was trained for.
    """
    metric_keys = list(results.keys())
    metric = metric_keys[2]  
    train_metric = results[metric]
    val_metric = results['val_' + metric]

    epochs_range = range(num_epochs)

    plt.figure(figsize=(10, 5))

    # Plotting metric
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_metric, label=f'Training {metric}')
    plt.plot(epochs_range, val_metric, label=f'Validation {metric}')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation {metric}')

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, results['loss'], label='Training Loss')
    plt.plot(epochs_range, results['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
def model_loader(model_path, results_path):
    
    with open(results_path, 'rb') as h:
        results = pickle.load(h)
        
    num_epochs = len(results['loss'])
    
    model = load_model(model_path)
    return results, model, num_epochs


def load_modeler(model_path, results_path, train=None, val=None):
    results, model, num_epochs = model_loader(model_path, results_path)
    load_viz(results,num_epochs)
    # Evaluate model
    train_scores = model.evaluate(train, verbose=0)
    val_scores = model.evaluate(val, verbose=0)
    
    print_results(results, train_scores, val_scores)


def print_results(results, train_scores, val_scores):
    # Display performance difference
    metric_list = list(results.keys())
    n = int(len(metric_list)/2)
    metrics_names = [metric_list[i] for i in range(0,n)]
    diff_scores = [val - train for train, val in zip(train_scores, val_scores)]
    performance_df = DataFrame([train_scores, val_scores, diff_scores], 
                               index=['Train', 'Val', 'Diff'], columns=metrics_names)
    display(performance_df)
    print('------------------------------\n')

def summary_viz(name, val_ds):
    base_dir = 'saved_models 2/'
    model_path = base_dir+name+'.keras'
    results_path = base_dir+name+'.pkl'
    results, model, num_epochs = model_loader(model_path, results_path)
    load_viz(results, num_epochs)
    val_scores = model.evaluate(val_ds, verbose=0)
    loss = round(val_scores[0],4)
    recall = round(val_scores[2],4)
    AUC = round(val_scores[1],4)
    name = 'baseline'
    df = pd.DataFrame([[loss,recall,AUC]],columns=['loss','recall','AUC'],index=[name])
    display(df)


def summary_df(file_names,model_names, val_ds):
    base_dir = 'saved_models 2/'
    score_list=[]
    for name in file_names:
        model_path = base_dir+name+'.keras'
        results_path = base_dir+name+'.pkl'
        results, model, num_epochs = model_loader(model_path, results_path)
        val_scores = model.evaluate(val_ds, verbose=0)
        loss = round(val_scores[0],4)
        recall = round(val_scores[2],4)
        AUC = round(val_scores[1],4)
        score_list.append([loss, recall, AUC])
    df = pd.DataFrame(score_list,columns=['loss','recall','AUC'],index=model_names)
    display(df)
    