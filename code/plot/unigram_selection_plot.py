__author__ = 'pierregagliardi'
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics

if __name__=="__main__":
    path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis/'
    folder = path+'pickle_unigrams/'
    unigrams=[100,1000,2000,3000,4000,5000,6000,7000,8000,9000]
    f1_score=[]
    accuracy=[]
    recall=[]
    for file in os.listdir(folder):
        #f1_score,accuracy,recall,precision
        y_true_y_pred=np.load(folder+file)
        y_true=y_true_y_pred[0]
        y_pred=y_true_y_pred[1]
        accuracy.append(metrics.accuracy_score(y_true,y_pred))
        f1_score.append(metrics.f1_score(y_true,y_pred))

    plt.plot(unigrams,accuracy,color='r',label='exactitude',marker='o')
    plt.plot(unigrams,f1_score,color='b',label='f1_score',marker='o')
    #plt.plot(unigrams,f1_score,color='g',label='recall',marker='o')
    plt.ylabel('accuracy')
    plt.xlabel('Number of unigrams')
    legend = plt.legend(loc='upper center')
    plt.grid(True)
    fig = plt.gcf()
    fig.savefig(path+'/selection_optimal_number_unigrams.png')
