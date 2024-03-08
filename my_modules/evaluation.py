from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def con_mat(model):
    y_preds = (model.predict(val_images) >= 0.5).astype(int)
    y_test = val_labels

    cnf_matrix = confusion_matrix(y_test,y_preds,normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.show()