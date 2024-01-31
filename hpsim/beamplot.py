"""
BeamPlot class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""
from typing import List
import logging
import pandas as pd
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl

from .constants import *


class BeamPlot():
    """An hpsim class for creating beam plots"""

    def __init__(self, nrow=1, ncol=1, hsize=None, vsize=None):
        """Creates and instance of a matplotlib figure

        Arguments:
           nrow (int): number of rows in figure plotting grid
           ncol (int): number of columns in figure plotting grid
           hsize (double): horizontal size (inches) of figure
           vsize (double): vertical size (inches) of figure
        """
        if hsize == None or vsize == None:
            self.fig = plt.figure(facecolor='white')
        else:
            self.fig = plt.figure(figsize = (hsize, vsize), facecolor='white')

        self.nrow = nrow
        self.ncol = ncol
        self.axes = [[0 for icol in range(nrow)] for irow in range(nrow)]
        print('{0:2} x {1:2} BeamPlot object created'.format(self.nrow, self.ncol))

    @classmethod
    def create_standard_plots(cls, beam, mask, title:str = "", 
                             rewrap_phase=None, figsize=(8,8)):
        """Create a standard 4x3 BeamPlot object. From top 1st to the 4th row,
        it plots
            iso_phase_space for xxp, yyp, phiw
            hist2d_phase_space for xxp, yyp, phiw
            profile for x, y, phi
            profile for xp, yp, w


        Arguments:
               beam (beam object): beam object containing coordinates to plot
               mask (Numpy vector, int): mask for filtering beam prior to plotting
               rewrap_phase(float, optional): rewrap phase by this number
               hsize(float, optional): horizontal size for the figure
               vsize(float, optional): vertical size for the figure

        Returns:
               BeamPlot object
        """
        plot = cls(nrow=4, ncol=3, hsize=figsize[0], vsize=figsize[1])
        plot.title(title)
        plot.iso_phase_space('xxp', beam, mask, 1)
        plot.iso_phase_space('yyp', beam, mask, 2)
        plot.iso_phase_space('phiw', beam, mask, 3, rewrap_phase=rewrap_phase)
        plot.hist2d_phase_space('xxp', beam, mask, 4)
        plot.hist2d_phase_space('yyp', beam, mask, 5)
        plot.hist2d_phase_space('phiw', beam, mask, 6, rewrap_phase=rewrap_phase)
        plot.profile('x', beam, mask, 7, 'g-')
        plot.profile('y', beam, mask, 8, 'g-')
        plot.profile('phi', beam, mask, 9, 'g-', rewrap_phase=rewrap_phase)
        plot.profile('xp', beam, mask, 10, 'g-')
        plot.profile('yp', beam, mask, 11, 'g-')
        plot.profile('w', beam, mask, 12, 'g-')
        plot.fig.tight_layout()
        return plot




    def title(self, title):
        """Place title string in window bar
        Arguments:
           title (str): figure title
        """
        man = plt.get_current_fig_manager()
        man.set_window_title(title)


    def clear(self):
        """Clear plot figure"""
        self.fig.clf()

    def hist1d(self, u_vals, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create 1d histogram of arbitrary vals in numpy array

        Arguments:
           u_vals (Numpy vector):values to plot
           nplt (int):which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (optional, [list of doubles]) [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False(default)-> linear plot

        """
        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_vals, nplt, nbins, xlabel, limits, norm, ylog)
        irow = (nplt-1)//self.ncol
        icol = (nplt-1)%self.ncol
        self.axes[irow][icol] = plt

        return

    def hist1d_coor(self, coor, beam, mask, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create a histogram style profile of beam coordinate

        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           beam (beam object): beam object containing coordinates to plot
           mask (numpy vector): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False-> linear plot
        """

        if coor in COORDINATES:
            u_index = COORDINATES.index(coor)
            u_label = COORDINATES[u_index]
            u_coor = beam.get_coor(u_label, mask)
            label = u_label + ' ' + USER_LABELS[u_label]

        elif coor in LOSSES:
            u_coor = 1.0*beam.get_coor('losses', mask)
            label = 'losses along beamline'

        if xlabel is not None:
            label = xlabel

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
        _plot_hist1d(plt, u_coor, nplt, nbins, label, limits, norm, ylog)
        irow = (nplt-1)//self.ncol
        icol = (nplt-1)%self.ncol
        self.axes[irow][icol] = plt

        return

    def profile(self, coor, beam, mask, nplt, marker='g-', nbins=50, 
                limits=None, ylog=False, rewrap_phase=None):
        """Create a profile of beam coordinate 

        Arguments:
           coor (str): coordinate to plot, either 'x', 'xp', 'y', 'yp', 'phi', 'w' or 'losses'
           beam (beam object): beam object containing coordinates to plot
           mask (numpy vector): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           nbins (int, optional): number of bins to plot
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
           ylog (logical, optional): True-> semilog plot, False-> linear plot
           rewrap_phase(int, optional): rewrap phase by this number
        """ 
        if coor in COORDINATES:
            u_index = COORDINATES.index(coor)
            u_label = COORDINATES[u_index]
            u_coor = beam.get_coor(u_label, mask)
            label = u_label + ' ' + USER_LABELS[u_label]
            if (u_label=="phi") and (rewrap_phase is not None):
                u_coor = u_coor%rewrap_phase

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_profile(plt, u_coor, marker, nbins, label, limits, ylog)
            irow = (nplt-1)//self.ncol
            icol = (nplt-1)%self.ncol
            self.axes[irow][icol] = plt

        return

    def phase_space(self, coor, beam, mask, nplt, marker='b,', limits=None):
        """Create beam phase space dot plot as nth subplot to figure

        Arguments:
           coor (str): text string either 'xxp', 'yyp' or 'phiw'
           beam (object): object containing beam to be plotted
           mask (Numpy array): mask for filter beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                        1 is upper left, nrow*ncol is lower right
           marker (str, optional): matplotlib color and marker, e.g. 'r.'
           limits (list of doubles, optional): plot limits [[xmin, xmax], [ymin, ymax]]
        """
        if coor in PHASESPACE:
            u_index = PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = COORDINATES[u_index]
            v_label = COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + USER_LABELS[u_label]
            labels[1] = v_label + ' ' + USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)
            _plot_phase_space(plt, u_coor, v_coor, marker, labels, limits)
            irow = (nplt-1)//self.ncol
            icol = (nplt-1)%self.ncol
            self.axes[irow][icol] = plt

        return
        
    def iso_phase_space(self, coor, beam, mask, nplt, nbins=50, rewrap_phase=None):
        """Create an isometric phase-space plot.

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins, respectively
           rewrap_phase(float, optional): rewrap phase by this number
        """
        if coor in PHASESPACE:
            u_index = PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = COORDINATES[u_index] 
            v_label = COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)
            if (u_label=="phi") and (rewrap_phase is not None):
                u_coor = u_coor%rewrap_phase
            if (v_label=="phi") and (rewrap_phase is not None):
                v_coor = v_coor%rewrap_phase   

            labels=['','']
            labels[0] = u_label + ' ' + USER_LABELS[u_label]
            labels[1] = v_label + ' ' + USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')
            _plot_iso_phase_space(plt, u_coor, v_coor, labels, nbins)
            irow = (nplt-1)//self.ncol
            icol = (nplt-1)%self.ncol
            self.axes[irow][icol] = plt

        return

    def surf_phase_space(self, coor, beam, mask, nplt, nbins=100, limits=None):
        """Create a surface phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """
        #nbins = 50
        if coor in PHASESPACE:
            u_index = PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = COORDINATES[u_index]
            v_label = COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)

            labels=['','']
            labels[0] = u_label + ' ' + USER_LABELS[u_label]
            labels[1] = v_label + ' ' + USER_LABELS[v_label]

            ax = self.fig.add_subplot(self.nrow, self.ncol, nplt, projection='3d')

            _plot_surf_phase_space(ax, u_coor, v_coor, labels, nbins, limits)
            irow = (nplt-1)//self.ncol
            icol = (nplt-1)%self.ncol
            self.axes[irow][icol] = ax

        return

    def hist2d(self, u_vals, v_vals, nplt, labels=None, nbins=100, limits=None):
        """Create an 2d histogram of user given u & v values

        Arguments:
           u_vals (Numpy vector):values to plot u-axis
           v_vals (Numpy vector):values to plot v-axis
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           labels ([str, str]): u- and v-axes lables
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
        """

        plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
        _plot_hist2d(plt, u_vals, v_vals, labels=labels, nbins=nbins, limits=limits)
        irow = (nplt-1)//self.ncol
        icol = (nplt-1)%self.ncol
        self.axes[irow][icol] = plt

        return

    def hist2d_phase_space(self, coor, beam, mask, nplt, nbins=100, 
                           limits=None, rewrap_phase=None):
        """Create an 2d histogram phase-space plot

        Arguments:
           coor (str): Phase space to plot, either 'xxp', 'yyp', or 'phiw' 
           beam (beam object): beam object containing coordinates to plot
           mask (Numpy vector, int): mask for filtering beam prior to plotting
           nplt (int): which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of x and y bins
           limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
           rewrap_phase(float, optional): rewrap phase by this number
        """
        if coor in PHASESPACE:
            u_index = PHASESPACE.index(coor) * 2
            v_index = u_index + 1
            u_label = COORDINATES[u_index]
            v_label = COORDINATES[v_index]
            u_coor = beam.get_coor(u_label, mask)
            v_coor = beam.get_coor(v_label, mask)
            if (u_label=="phi") and (rewrap_phase is not None):
                u_coor = u_coor%rewrap_phase
            if (v_label=="phi") and (rewrap_phase is not None):
                v_coor = v_coor%rewrap_phase    
        
            labels = ['', '']
            labels[0] = u_label + ' ' + USER_LABELS[u_label]
            labels[1] = v_label + ' ' + USER_LABELS[v_label]

            plt = self.fig.add_subplot(self.nrow, self.ncol, nplt)          
            _plot_hist2d(plt, u_coor, v_coor, labels=labels, nbins=nbins, limits=limits)
            irow = (nplt-1)//self.ncol
            icol = (nplt-1)%self.ncol
            self.axes[irow][icol] = plt

        return

    def draw(self):
        """Draw figure. Used in interactive mode"""
        plt.draw()
        return

    def show(self):
        """Show the plots. Used in non-interactive mode"""
        plt.tight_layout()
        plt.show()
        return




################################################################################
#
# private  functions
#

def _get_labels(labels):
    """Get axis labels for plotting
    Arguments:
       labels([str, str]): list of x, y-axis labels
    """
    if labels == None:
        u_label = 'x-axis'
        v_label = 'y-axis'
    elif isinstance(labels, list):
        u_label = labels[0]
        v_label = labels[1]
    else:
        u_label = ''
        v_label = ''
    return (u_label, v_label)

def _get_plimits(limits, u_coor, v_coor):
    """Returns the xmin, xmax, ymin, ymax for plot range.

    Private method

    Arguments:
       plim (list of doubles): [[xmin, xmax],[ymin, ymax]] or 
                               [[xmin, xmax],[]] or [[], [ymin, ymax]] or 
                               [ymin, ymax] or None
       u_coor (Numpy vector doubles): xcoordinates
       v_coor (Numpy vector doubles): ycoordinates

    Returns:
       list (double): [[xlo, xup],[ylo, yup]] containing the x and y limits
                      for the plotting range
    """
    if limits not in [None, []]:
        if isinstance(limits[0], list):
            #list of lists [[xmin, xmax],[ymin, ymax]]
            if limits[0] != []:
                min_x = limits[0][0]
                max_x = limits[0][1]
            else:
                min_x = min(u_coor)
                max_x = max(u_coor)
                if min_x == max_x:
                    min_x = u_coor *0.9
                    max_x = u_coor *1.1

            if limits[1] != []:
                min_y = limits[1][0]
                max_y = limits[1][1]
            else:
                min_y = min(v_coor)
                max_y = max(v_coor)
                if min_y == max_y:
                    min_y = v_coor *0.9
                    max_y = v_coor *1.1

        else:
            # list of y-values only, [ymin, ymax]
            min_x = min(u_coor)
            max_x = max(u_coor)
            min_y = limits[0]
            max_y = limits[1]
    else:
        min_x = min(u_coor)
        max_x = max(u_coor)
        min_y = min(v_coor)
        max_y = max(v_coor)
        if min_x == max_x:
            min_x = min_x *0.9
            max_x = max_x *1.1
        if min_y == max_y:
            min_y = min_y *0.9
            max_y = max_y *1.1

    return (min_x, max_x), (min_y, max_y)


def _plot_hist1d(plt, u_vals, nplt, nbins=50, xlabel=None, limits=None, norm=1.0, ylog=False):
        """Create 1d histogram of arbitrary vals in numpy array

        Private method

        Arguments:
           plt (Pyplot figure subplot object): figure to place subplot in
           u_vals (Numpy vector):values to plot
           nplt (int):which plot in figure grid, by row, 
                       1 is upper left, nrow*ncol is lower right
           nbins (int, optional): number of bins to plot
           xlabel (str, optional): x-axis label
           limits (list of doubles, optional): [[xmin, xmax], [ymin, ymax]]
           norm (double, optional): normalization factor for plot
           ylog (logical, optional): True-> semilog plot, False(default)-> linear plot

        """
        wghts = (1.0 / norm) * np.ones(len(u_vals))
        hist, bins, patches = plt.hist(u_vals, bins=nbins, weights=wghts, log=ylog)
        xrng, yrng = _get_plimits(limits, bins, hist)
        if xlabel is None:
            xlabel = 'variable'
        plt.set_xlabel(xlabel)
        plt.set_ylabel('counts/bin')
        if limits is not None:
            plt.set_xlim(xrng)
            if ylog is False:
                plt.set_ylim(yrng)
        return

def _plot_profile(plt, u_vals, marker='g-', nbins=50, label=None, limits=None, ylog=False):
    """Create a profile of beam coordinate 

    Arguments:
       plt (Pyplot plot object): plot figure object
       u_vals (Numpy vector, double): x-coordinates of data to be plotted
       marker (str, optional): matplotlib color and marker, e.g. 'r.'
       nbins (int, optional): number of bins to plot
       label (str): u-axis label
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]
       ylog (logical, optional): True-> semilog plot, False-> linear plot
    """ 
    hist, bins = np.histogram(u_vals, bins = nbins)
    xrng, yrng = _get_plimits(limits, bins, hist)

    plt.set_xlabel(label)
    plt.set_ylabel('counts/bin')
    if ylog:
        plt.semilogy(bins[:-1], hist, marker)
        plt.set_xlim(xrng)
        if not (limits == None or limits[1] == []):
            plt.set_ylim(yrng)
    else:
        plt.plot(bins[:-1], hist, marker)
        plt.set_xlim(xrng)
        plt.set_ylim(yrng)

    return

def _plot_phase_space(plt, u_coor, v_coor, marker='b,', labels=None, limits=None):
    """Create beam phase space dot plot as nth subplot to figure

    Arguments:
       plt (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       marker (str, optional): matplotlib color and marker, e.g. 'r.'
       labels ([str, str]): u- and v-axes lables 
       limits (list of doubles, optional): plot limits [[xmin, xmax], [ymin, ymax]]
    """
    u_label, v_label = _get_labels(labels)
    xrng, yrng = _get_plimits(limits, u_coor, v_coor)
    plt.set_xlabel(u_label)
    plt.set_ylabel(v_label)
    plt.plot(u_coor, v_coor, marker)
    plt.set_xlim(xrng)
    plt.set_ylim(yrng)
    return

def _plot_iso_phase_space(ax, u_coor, v_coor, labels=None, nbins=50):
    """Create an isometric phase-space plot.

    Arguments:
       ax (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       labels ([str, str]): u- and v-axes lables 
       nbins (int, optional): number of x and y bins, respectively
    """

    u_label, v_label = _get_labels(labels)
    limits=None
    [min_x, max_x], [min_y, max_y] = _get_plimits(limits, u_coor, v_coor)

    #TEMP
    print([min_x, max_x], [min_y, max_y])

    ps_histo, u_bins, v_bins = np.histogram2d(y=v_coor, x=u_coor, bins=nbins, \
                                              range = [[min_x, max_x], [min_y, max_y]])
    # when plotting over a large range and the peak is at the edge, then find which
    # edge and add and extra bin there to ensure it is plotted. This is to overcome
    # a bug in matplotlib polycollection



    hist_max = np.max(ps_histo)
    hist_shape = np.shape(ps_histo)
    loc_max = np.unravel_index(np.argmax(ps_histo),np.shape(ps_histo))
    if hist_max > 0:
        if loc_max[0] == 0 or loc_max[-1] == hist_shape[0]-1: 
            if loc_max[0] == 0:
                min_x = min_x - (u_bins[1] - u_bins[0])
            #elif loc_max[-1] == hist_shape(ps_histo)-1:
            elif loc_max[-1] == hist_shape[0]-1:
                max_x = max_x + (u_bins[1] - u_bins[0])
            
            new_bins = nbins+1
            ps_histo, u_bins, v_bins = np.histogram2d(y=v_coor, x=u_coor, bins=new_bins, \
                                              range = [[min_x, max_x], [min_y, max_y]])

    ps_histo = ps_histo.T
    verts = []
    for v_slice in ps_histo:
        v_slice[0] = 0.
        v_slice[-1] = 0.
        verts.append(list(zip(u_bins, v_slice)))

    poly = PolyCollection(verts, closed=False)

    from matplotlib import cm
#            m = cm.ScalarMappable(cmap=cm.jet)
#            m.set_clim(vmin=0, vmax=100)
    poly.set_cmap(cm.jet)
    poly.set_clim(vmin=0, vmax=100)
    poly.set_color('lightgreen')
#            poly.set_color('blue')
    poly.set_edgecolor('gray')
#            poly.set_edgecolor('white')
    poly.set_alpha(0.75)

#ljr
# added new lines to fix problem between mpl v1.1.1 and v 2.1.1
    if int(mpl.__version__.split('.')[0]) > 1:
        v_bins_new = v_bins[0:-1]
    else:
        v_bins_new = v_bins
#ljr
    ax.add_collection3d(poly, zs=v_bins_new, zdir='y') #ljr

    ax.set_xlabel(u_label)
    ax.set_xlim3d(u_bins[0],u_bins[-1])
    ax.set_ylabel(v_label)
    ax.set_ylim3d(v_bins[0],v_bins[-1])
    ax.set_zlabel('Amplitude [counts]')
    ax.set_zlim3d(0, np.max(ps_histo))

    return

def _plot_surf_phase_space(ax, u_coor, v_coor, labels=None, nbins=100, limits=None):
    """Create a surface phase-space plot

    Arguments:
       ax (Pyplot plot object): plot figure object
       u_coor (Numpy vector, double): x-coordinates of data to be plotted
       v_coor (Numpy vector, double): y-coordinates of data to be plotted
       labels ([str, str]): u- and v-axes lables
       nbins (int, optional): number of x and y bins
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
    """

    u_label, v_label = _get_labels(labels)
    [min_x, max_x], [min_y, max_y] = _get_plimits(limits, u_coor, v_coor) 

    dx = (max_x - min_x)/float(nbins)
    dy = (max_y - min_y)/float(nbins)

    Xr = np.arange(min_x, max_x, dx)
    Yr = np.arange(min_y, max_y, dy)
    Xr, Yr = np.meshgrid(Xr, Yr)

    ps_histo, u_bins, v_bins = np.histogram2d(y=v_coor, x=u_coor, bins=nbins, \
                                              range = [[min_x, max_x], [min_y, max_y]],\
                                              density=False) #True)
    ps_histo = ps_histo.T

    ZMAX = float(np.max(ps_histo))
    colors = [[(0,0,0,0) for _ in range(nbins)] for _ in range(nbins)]
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colormap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    for y in range(nbins):
        for x in range(nbins):
            if ps_histo[x, y] == 0:
                colors[x][y] = (1,1,1,1)
            else:
                colors[x][y] = colormap.to_rgba(ps_histo[x,y]/ZMAX)

    ps_surf = ax.plot_surface(Xr, Yr, ps_histo, rstride=1, cstride=1, linewidth=0, facecolors=colors)

    ax.set_xlabel(u_label)
    ax.set_xlim3d(u_bins[0], u_bins[-1])
    ax.set_ylabel(v_label)
    ax.set_ylim3d(v_bins[0], v_bins[-1])
    ax.set_zlabel('Amplitude [counts]')
    ax.set_zlim3d(0, np.max(ps_histo))
    return

def _plot_hist2d(axPS, u_vals, v_vals, labels=None, nbins=100, limits=None):
    """Create an 2d histogram of user given u & v values

    Arguments:
       axPS (Pyplot figure object): plot object in figure
       u_vals (Numpy vector):values to plot u-axis
       v_vals (Numpy vector):values to plot v-axis
       labels ([str, str]): u- and v-axes lables
       nbins (int, optional): number of x and y bins
       limits (list, doubles, optional): [[xmin, xmax], [ymin, ymax]]       
    """
    u_label, v_label = _get_labels(labels)

    # the scatter plot:

    [min_x, max_x], [min_y, max_y] = _get_plimits(limits, u_vals, v_vals) 

    ps_histo, u_bins, v_bins = np.histogram2d(y=v_vals, x=u_vals, bins=nbins, \
                                              range = [[min_x, max_x], [min_y, max_y]],\
                                              density=True)
    ps_histo = ps_histo.T

    # mask zeros so they are not plotted
    ps_histo_masked = np.ma.masked_where(ps_histo == 0, ps_histo)

    extent = [min_x, max_x, min_y, max_y]
    figasp = float(max_x - min_x)/float(max_y - min_y)

    axPS.set_xlabel(u_label)
    axPS.set_ylabel(v_label)
    axPS.imshow(ps_histo_masked, cmap='jet', extent=extent,interpolation='none', origin='lower', \
                    aspect=figasp)

    return


