"""
PlotData class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
Note that plotdata is the class to get the beam parameters at certain
points that are under "monitor".
"""
import logging
import numpy as np
import pandas as pd

import HPSim
from .constants import *

class PlotData():
    """An hpsim class for storing data at certain monitored points.

    Usage:
        n_monitor = bl.get_monitor_numbers(SIM_START, SIM_STOP)
        plotdata = hpsim.PlotData(n_monitor)
    """
    _plotdata = ""
    _saved_vars = ["xavg", "xsig", "xpavg", "xpsig", "xemit",
                  "yavg", "ysig", "ypavg", "ypsig", "yemit",
                  "phiavg", "phisig", "phiref", "zemit",
                  "wavg", "wsig", "wref",
                  "loss_ratio", "loss_local", "model_index"]
    def __init__(self, size: int):
        """
        Arguments:
           size(int): size of plot data
        Returns:
           PlotData object
        """
        self._plotdata = HPSim.PlotData(size)
        PlotData._plotdata = self._plotdata

    @property
    def vars(self):
        """Get list of variables saved by PlotData.
        Returns:
        list of saved var names
        """
        return list(self._saved_vars)
    
    def get_values(self, option):
        """
        option(str): axis to choose from in 'xavg', 'xsig', 'xpavg', 'xpsig', 'xemit',
                                      'yavg', 'ysig', 'ypavg', 'ypsig', 'yemit',
                                      'phiavg', 'phisig', 'phiref', 
                                      'wavg', 'wsig', 'zemit',
                                      'wref', 'loss_ratio', 'loss_local',
                                      'model_index'
        Returns:
        np.array
        """
        if option not in self._saved_vars:
            logging.warning(f"{option} is not available for plotdata")
            return None
        values = self._plotdata.get_values(option)
        if option in ["model_index"]:
            #print(values)
            values = np.int_(values)
        else:
            values *= USER_UNITS[option]
        return values

    def update_beamline_dataframe(self, dfbl):
        """
        Update all the plotdata values to the beamline dataframe
        Arguments:
           dfbl(pd.DataFrame): df returned from Beamline.data
        """
        idxs = self.get_values("model_index") # obtain model_index (idx)
        _locations = np.where(idxs>0)[0]
        if len(_locations)==0:
            logging.error("no plot data has been recorded.")
            return dfbl
        last_idx = _locations[-1]+1 # if idx ==0, it is not filled.
        idxs = idxs[:last_idx]
        for var in self.vars: # go through each vars, like "xavg", "xsig"...
            if var == "model_index": # already got this
                continue
            if var not in dfbl.columns:
                dfbl[var] = np.nan #initialize to nan if not exist
            # create a series that has column = name, and index = idx
            vals =pd.Series(self.get_values(var)[:last_idx], name=var, index=idxs)
            # update the values to dfbl. If vals are nan, it won't replace.
            dfbl.update(vals)
        return dfbl

    @classmethod
    def reset(self):
        """ reset the plotdata values to zeros.
        """
        self._plotdata.reset()