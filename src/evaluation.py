import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime.lime_image import LimeImageExplainer

def evaluate_model(model, dataset):
    """
    Evaluate the model on a given dataset.

    Parameters:
    - model: The machine learning model to be evaluated.
    - dataset: The dataset to evaluate the model on.

    Returns:
    - A list of rounded scores: [loss, accuracy, auc].
    """
    test_scores = model.evaluate(dataset, verbose=0)
    scores = [round(test_scores[i], 4) for i in [0, 2, 1]]
    return scores

def extract_images_and_labels(dataset):
    """
    Extract images and labels from the dataset.

    Parameters:
    - dataset: A Tensorflow dataset

    Returns:
    - A tuple containing arrays of images and labels.
    """
    images = np.concatenate([img_batch.numpy() for img_batch, _ in dataset], axis=0)
    labels = np.concatenate([lbl_batch.numpy() for _, lbl_batch in dataset], axis=0).squeeze()
    return images, labels

def analyze_predictions(model, images, labels):
    """
    Analyze predictions made by the model.

    Parameters:
    - model: The trained model.
    - images: Numpy array of images.
    - labels: Numpy array of true labels.

    Returns:
    - Indices of best true positives and true negatives.
    """
    probabilities = model.predict(images, verbose=0).squeeze()
    predictions = np.round(probabilities)
    true_pos, true_neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    pos_preds, neg_preds = np.where(predictions == 1)[0], np.where(predictions == 0)[0]
    tp, tn = np.intersect1d(true_pos, pos_preds), np.intersect1d(true_neg, neg_preds)
    best_tp = tp[np.argsort(-probabilities[tp])]
    best_tn = tn[np.argsort(probabilities[tn])]
    return best_tp, best_tn

def LIME_viz(model, dataset, type='TP'):
    """
    Displays the best predicted images including LIME explanations.

    Parameters:
    - model: The model to make predictions.
    - dataset: The dataset containing images and labels.
    - highlight_type: Type of predictions to highlight ('TP' or 'TN').

    Displays:
    - A 3x3 subplot of highlighted images based on the prediction type.
    """
    def make_prediction(image):
        gray_image = rgb2gray(image).reshape(-1, 64, 87, 1)
        probability = model.predict(gray_image, verbose=0)
        return np.hstack([1 - probability, probability])

    images, labels = extract_images_and_labels(dataset)
    best_tp, best_tn = analyze_predictions(model, images, labels)
    indices = best_tp if type == 'TP' else best_tn
    case = 'Pneumonia' if type == 'TP' else 'Normal'

    segment_image = SegmentationAlgorithm('slic', n_segments=75, compactness=10, sigma=1)
    explainer = LimeImageExplainer(random_state=123)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle('Top 9 Correctly Predicted '+case+' Cases', fontsize=16, y=0.92)  # Add a title to the figure
    
    for idx, ax in zip(indices[:9], axes.ravel()):  # Loop through the first 9 indices and axes
        explanation = explainer.explain_instance(
            gray2rgb(images[idx].squeeze()), make_prediction,
            random_seed=123, segmentation_fn=segment_image
        )
        img, mask = explanation.get_image_and_mask(labels[idx], positive_only=False, hide_rest=False)
        ax.imshow(mark_boundaries(img, mask))
        ax.axis('off')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    # for idx, ax in zip(indices[:9], axes.ravel()):  # Limit to first 9 images
    #     explanation = explainer.explain_instance(
    #         gray2rgb(images[idx].squeeze()), make_prediction,
    #         random_seed=123, segmentation_fn=segment_image
    #     )
    #     img, mask = explanation.get_image_and_mask(labels[idx], positive_only=False, hide_rest=False)
    #     ax.imshow(mark_boundaries(img, mask))
    #     ax.axis('off')
    # plt.tight_layout()
    plt.savefig('images/'+type+'_LIME_figures.png')
    plt.show()