from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def con_mat(model,test_images, test_labels):
    y_preds = (model.predict(val_images) >= 0.5).astype(int)
    y_test = val_labels

    cnf_matrix = confusion_matrix(y_test,y_preds,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.show()

def model_evaluator(model, test_ds):
    scores = []
    test_scores = model.evaluate(test_ds, verbose=0)
    loss = round(test_scores[0],4)
    recall = round(test_scores[2],4)
    AUC = round(test_scores[1],4)
    scores.append([loss, recall, AUC])
    return scores
