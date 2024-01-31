"""
created by En-Chuan Huang on 12/17/2022.
"""
import numpy as np
import pandas as pd
import logging

class LossDetectors(object):
    """ A helper class to calculate losses along the linac. At this point,
    we are assuming each detector is identical. For LANSCE, it's APs.
    """

    def __init__(self, df:pd.DataFrame):
        """initialize loss detectors

        Arguments:
        df(DataFrame): columns must contain `name`, `z` for location along the 
                        linac, and `x` for perpendicular locations.
        """
        complete = True
        for col in ["name", "z", "x"]:
            if col not in df.columns:
                logging.error(f"{col} not in imported DataFrame")
                complete = False
        if complete ==False:
            logging.error("Cannot initialize LossDetectors")
            return
        df = df.copy()
        df["signal"] = 0
        self._df = df

    
    def get_loss_aperture_by_z(self, zAP, z, w, gamma=3.45):
        """
        Calculate relative loss signal at specific zAP with particles lost at 
        `z` with energy `w`.
        
        Parameters:
        zAP(float)   : z locations of the AP
        z(np.ndarray): z locations of the lost particles
        w(np.ndarray): energy of the lost particles
        gamma(float) : horizontal distance with respect to the linac
        """
        gamma_term = (gamma/2.)**2
        zdiffSq = (z-zAP)**2
        A = -1.989e-1 - 1.967e-3*w + 1.201e-4*(w**2) -5.165e-8*(w**3)
        sigAP = np.sum(A / (zdiffSq + gamma_term))
        return sigAP


    def calculate_loss_aperture(self, z, w):
        """
        Calculate relative loss signal from particle losses provided by HPSim.
        
        Parameters:
        z(np.ndarray): z locations of the lost particles
        w(np.ndarray): energy of the lost particles
        """
        zAPs = self._df["z"]
        sigAP = np.zeros_like(zAPs)
        gamma_term = (self._df["x"]/2.)**2
        for idx, row in self._df.iterrows():
            zAP = row["z"]
            gamma = row["x"]
            self._df.loc[idx, "sig_aperture"] = \
                        self.get_loss_aperture_by_z(zAP, z, w, gamma)

    def sum_loss(self):
        """
        Calculate loss from each term
        """
        self._df["signal"] = 0
        if "sig_aperture" in self._df.columns:
            self._df["signal"] += self._df["sig_aperture"]
        if "sig_intrabeam" in self._df.columns:
            self._df["signal"] += self._df["sig_intrabeam"]
        if "sig_residual_gas" in self._df.columns:
            self._df["signal"] += self._df["sig_residual_gas"]
        if "sig_lorentz" in self._df.columns:
            self._df["signal"] += self._df["sig_lorentz"]

    @property
    def data(self):
        """ Returns a copy of current DataFrame """
        return self._df.copy()
