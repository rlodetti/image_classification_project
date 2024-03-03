import time
import matplotlib.pyplot as plt
from pandas import DataFrame 
from tensorflow.keras import models, layers, metrics, callbacks, regularizers

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

def modeler(model, train=train_ds, val=val_ds, metrics=METRICS, optimizer='adam', num_epochs=10, early_stopping=None):
    """
    Compiles, trains, and evaluates the model, and visualizes training results.

    Parameters:
    - model: The Keras model to be trained.
    - metrics: List of metrics to be evaluated by the model during training and testing.
    - optimizer: String (name of optimizer) or optimizer instance to use.
    - num_epochs: Number of epochs to train the model.
    - early_stopping: EarlyStopping callback to stop training early if no improvement.
    - train_ds: Training dataset.
    - val_ds: Validation dataset.
    """
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    start_time = time.time()
    callbacks_list = [early_stopping] if early_stopping else []
    results = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=0,
                        callbacks=callbacks_list)
    end_time = time.time()

    print(f"Training time: {end_time - start_time} seconds\n")
    visualize_training_results(results, num_epochs=num_epochs)

    # Evaluate model
    train_scores = model.evaluate(train_ds, verbose=0)
    val_scores = model.evaluate(val_ds, verbose=0)
    
    # Display performance difference
    metrics_names = ['loss'] + [metric.name for metric in metrics]
    diff_scores = [val - train for train, val in zip(train_scores, val_scores)]
    performance_df = DataFrame([train_scores, val_scores, diff_scores], index=['Train', 'Val', 'Diff'], columns=metrics_names)
    display(performance_df)
    print('------------------------------\n')

    return results, model