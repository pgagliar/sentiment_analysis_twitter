__author__ = 'pierregagliardi'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, path, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['positive','negative','neutral']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = plt.gcf()
    fig.savefig(path+'/confusion.png')

    if __name__=="__main__":
     path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/pickle_unigrams_ytrue_ypred/'
     ytrue_ypred=np.load(path+'60000_all_features_1000.pkl')
     cm = confusion_matrix(ytrue_ypred)
     plot_confusion_matrix(cm, path)