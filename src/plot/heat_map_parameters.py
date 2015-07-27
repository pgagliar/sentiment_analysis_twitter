__author__ = 'pierregagliardi'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from itertools import product
# Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_heat_map(scores, gamma_range, C_range,path_to_Figures):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.3,vmax=0.62, midpoint=0.55))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    fig = plt.gcf()
    fig.savefig(path+'/heat_map_without_varThres_1000.png')

if __name__=="__main__":
    path='/Users/pierregagliardi/DossierTravail/Programmation/PythonPath/projet_sentiment_analysis'
    scores=np.load(path+'/pickle_hyper_parameters/60000_all_features_1000_rbf_without_varThres.pkl')
    C_range = np.logspace(0, 3, 4)
    gamma_range = np.logspace(-3, -1, 3)
    list=list(product(C_range,gamma_range))
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    plot_heat_map(scores,gamma_range,C_range,path+'/Figures')
