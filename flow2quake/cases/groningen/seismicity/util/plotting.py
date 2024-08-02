# ========================================================================================================================================================================
# ========================================================================================================================================================================
# ======================================================================================================== Plotting Functions =============================================
# ========================================================================================================================================================================
# ========================================================================================================================================================================

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
font = {'family' : 'sans-serif',
        'size'   : 18}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 22})
import seaborn as sns

# ===================================================================== Classic cornerplot =================================================================

def cornerplot(SF,FILENAME):
    betas = SF.Training['betas']
    fig = plt.figure(figsize=(17,15))
    gs  = GridSpec(betas.shape[-1], betas.shape[-1], figure=fig)
    for ii in range(betas.shape[-1]):
        for jj in range(betas.shape[-1]):
            if jj >= ii:
                continue
            ax = fig.add_subplot(gs[ii,jj])
            sns.kdeplot(x=betas[:,ii],y=betas[:,jj],ax=ax,cmap="Greens")
            ax.set_xlabel('beta{}'.format(ii))
            ax.set_ylabel('beta{}'.format(jj))
    plt.locator_params(nbins=3)
    plt.tight_layout()
    plt.savefig(FILENAME, bbox_inches='tight')

# ===================================================================== Classic rate plot =================================================================
def rateplot(SF,FILENAME):
    # -- Training 
    fig,ax = plt.subplots(1, figsize=(15, 8))
    Rpred   = SF.FullModel['Rpred']
    # --  logn : normalization of logp
    logn    = SF.Training['logp']-np.nanmin(SF.Training['logp'])
    logn    = logn/np.nanmax(logn)
    logpn   = 1 - logn
    idx     = np.argsort(logpn)[::-1] #argsort returns the indexes to sort the table
    Rpred_t = np.repeat(SF.FullModel['Time'][:-1],2)[1:] #each value is repeated twice to plot the yearly values as segments

    #iterate on increasing logpn
    for ii in idx:
      #plot with different grey shades related to the likelihood
      ax.plot(Rpred_t,Rpred[ii,:].repeat(2)[:-1],linewidth=1.0,color='{}'.format(logpn[ii]),alpha=0.05)

    #blue plot is the maximum likelihood model == MAP
    ax.plot(Rpred_t,Rpred[np.argmin(SF.Training['logp']),:].repeat(2)[:-1],linewidth=2.0,color='b', label='MAP model')

    #red plot is the observed seismicity catalog
    Robs   = SF.FullModel['Robs']
    Robs_t = np.repeat(SF.Catalogue['Dates'],2)[1:]
    ax.plot(Robs_t,Robs.repeat(2)[:-1],'r--', label='historical data')

    #vertical blue lines denote the training period
    ax.axvline(SF.training_period[0], label='history matching period',c='darkblue')
    ax.axvline(SF.training_period[1],c='darkblue')

    #vertical red lines denote the training period
    ax.axvline(SF.validation_period[0], label='testing period',c='darkred')
    ax.axvline(SF.validation_period[1],c='darkred')

    ax.set_xlim([SF.training_period[0]-1,SF.validation_period[1]+1])
    ax.set_ylabel('Earthquake rate')
    ax.set_xlabel('Time')
    #ax.set_title('Training med(logp)={:.3f}, Validation med(logp)={:.3f}'.format(np.median(SF.Training['logp']),np.median(SF.Validation['logp'])))
    ax.set_title('Training med(logp)={:.3f}'.format(np.median(SF.Training['logp'])))
    ax.legend(loc='upper left')

    plt.savefig(FILENAME, bbox_inches='tight',facecolor='w')

# ========================================================================================================================================================================
# ===================================================================== Different types of Map plots =================================================================
# ========================================================================================================================================================================

def mapplot(SF,FILENAME):
    """Plots the seismicity rate at the last value of coulomb stress in time
    Corresponds to the derivative of the failure function in every mesh point"""
    # --- Plotting the Spatial Map ---
    Rplot   = SF.ForecastingModel.Rates(SF.Training['betas'][np.argmin(SF.Training['logp']),:])
    fig,axs = plt.subplots(1,figsize=(15,15))
    p1      = axs.add_patch(Polygon(np.array(RES['Outline'][['X','Y']]/1000),fill=False))
    quad0   = axs.pcolormesh(RES['X']/1000,RES['Y']/1000,Rplot[:,:,-1],vmin=0.0,vmax=np.nanmax(Rplot[:,:,-1]),clip_path=p1,clip_on=True,cmap='inferno'); 
    cb0     = plt.colorbar(quad0,ax=axs,label='Events per m^2',orientation='horizontal')
    axs.set_aspect('equal')
    axs.axis('off')
    axs.set_xlim([np.min(RES['Outline']['X'])/1000,np.max(RES['Outline']['X'])/1000])
    axs.set_ylim([np.min(RES['Outline']['Y'])/1000,np.max(RES['Outline']['Y'])/1000])
    plt.savefig(FILENAME, bbox_inches='tight',facecolor='w')




