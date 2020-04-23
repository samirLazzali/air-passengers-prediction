import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
from matplotlib import gridspec
from scipy.stats import describe
import numpy as np

def stats_histo(data,name="--", ratio=2):
    """
    Affiche la distributions et les statistiques descriptives.
    """

    jbv, jbp = np.round(jarque_bera(data),3)
    obs, minmax, mean, var, skew, kuro = describe(data)

    std = np.round(np.sqrt(var),3)
    skew = np.round(skew,3)
    kuro = np.round(kuro,3)
    mean = np.round(mean,3)
    minmax = np.round(minmax,3)
    median = np.round(np.median(data),3)

    fig = plt.figure(figsize=(10, 4)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[ratio, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.hist(data,bins=26, linestyle = 'solid', edgecolor ="#c0c0c0", color="#6e8fd0");
    ax1 = plt.subplot(gs[1])
    ax1.plot()
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.annotate("Série : {}".format(name),[0.01,0.97],size='x-large',family='serif')
    ax1.annotate("Observations : {}".format(obs),[0.01,0.87],size='x-large',family='serif')
    ax1.axhline(y=.80, xmin=0.80, xmax=0.20, c='black', lw=0.7)
    ax1.annotate("Moyenne : {}".format(mean),[0.01,0.70],size='x-large',family='serif')
    ax1.annotate("Max : {}".format(minmax[0]),[0.01,0.60],size='x-large',family='serif')
    ax1.annotate("Min : {}".format(minmax[1]),[0.01,0.50], size='x-large', family='serif')
    ax1.annotate("Std. Dev : {}".format(std),[0.01,0.40],size='x-large',family='serif')
    ax1.annotate("Skewness : {}".format(skew),[0.01,0.30],size='x-large',family='serif')
    ax1.annotate("Kurtosis : {}".format(kuro),[0.01,0.20], size='x-large', family='serif')
    ax1.annotate("Médiane : {}".format(median),[0.01,0.10],size='x-large',family='serif')
    ax1.annotate("P-valeur : {}".format(jbp),[0.01,0.00], size='x-large', family='serif')
    plt.tight_layout()