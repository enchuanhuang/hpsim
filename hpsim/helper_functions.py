"""
Helper functions
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""

import sys
import os
import shutil
from typing import List
import logging
from typing import Optional

#import various packages
import numpy as np
import pandas as pd

import math

# HPSim related ones
import HPSim as HPSim
from . import lcsutil as lcs
from .constants import *
from .beamline import BeamLine
from .dbconnection import DBConnection
from .spacecharge import SpaceCharge


def set_gpu(n):
    """Set which GPU to use
    Arguments:
       n(int): number of GPU to use
       
    """
    HPSim.set_gpu(n)

def set_log_level(level):
    """Set logging level for the C++/CUDA part.
    Arguments:
       level(int): 0=fatal, error, warning, info, and debug
    """
    HPSim.set_log_level(level)

def set_debug_level(level):
    """Set deubugging level for the C++/CUDA part.
    Arguments:
       level(int): higher number has more detail
    """
    HPSim.set_debug_level(level)


def most_freq_value(vals):
    """ Returns an estimate of the most frequently occuring value
    by first histograming the the Numpy array npvals in a histogram
    with unit bins, then finding the peak, then averaging that along 
    with the adjacent bins to get an estimate of the value that 
    represents the most frequency value.

    Arguments:
       vals(Numpy array): input 1D array

    Returns:
       estimate of the value that occurs most frequently

    """
    minval = min(vals)
    maxval = max(vals)
    bins = max(int(maxval-minval) + 1, 3)
    hist, bin_edge = np.histogram(vals, bins=bins, density=False)
    nbins = len(hist)
    bin_width = bin_edge[1] - bin_edge[0]
    hist_max = np.amax(hist)
    bin_max_indx = np.argmax(hist)
    if bin_max_indx == 0:
        pk_bin_avg = bin_edge[bin_max_indx]
    elif bin_max_indx > 0 and bin_max_indx < nbins:
        ll = bin_max_indx - 1
        ul = bin_max_indx + 2
        pk_bin_avg = np.average(bin_edge[ll:ul], weights=hist[ll:ul])
    else:
        pk_bin_avg = bin_edge[bin_max_indx]

    return pk_bin_avg + 0.5 * bin_width

def modulo_phase(phase_dist, ref_phs):
    """Return the phase coordinates of beam after modulo 360 deg wrt ref_phs 
       has been applied

    Arguments:

       phase_dist (Numpy vector, doubles): phase coordinates (deg)
       ref_phs (double): reference phase for modulo calc

    """

    return ((phase_dist - ref_phs + math.pi * DEG) % (2*math.pi * DEG)) + ref_phs - (math.pi * DEG)



def set_db_epics(pv_name, value):
    """Change EPICS PV value in database

    Arguments:

       pv_name(str): EPICS pv name string
       value(str or double): value to set Epics PV to 
       Note: DBConnection and BeamLine must be already be established
    """
    HPSim.set_db_epics(pv_name, str(value), DBConnection.dbconnection,
                     BeamLine.beamline)

def get_db_epics(pv_name):
    """Retrieve EPICS PV value in database

    Arguments:
       pv_name (str): EPICS pv name string
       Note: DBConnection must be already be established
    """
    value = HPSim.get_db_epics(pv_name, DBConnection.dbconnection)
    if lcs.get_pv_type(pv_name) != 'L':
        value = float(value)
    return value
    
def set_db_model(table_name, field_name, value):
    """Change model database parameter value given by table_name and field_name
    
    Arguments:
       table_name (str): name of element in db table
       field_name (str): name of field to change of element in table
       value (str or double): value to set db field to 
       Note: DBConnection and BeamLine must be already be established
    """
    HPSim.set_db_model(table_name, field_name, str(value), DBConnection.dbconnection,
                     BeamLine.beamline)

def get_db_model(elem_name, field_name):
    """Retrieve model database parameter value given by table_name and field_name

    Arguments:
       table_name (str): name of element in db table
       field_name (str): name of field to change of element in table
       Note: DBConnection and BeamLine must be already be established
    """
    text_fields = ['id', 'name', 'model_type']
    value = HPSim.get_db_model(elem_name, field_name, DBConnection.dbconnection)
    if field_name not in text_fields:
        #convert to float
        value = float(value)
    return value
        
def get_element_list(start_elem_name = "", end_elem_name = "", elem_type = ""):
    return BeamLine.get_element_list(start_elem_name, end_elem_name, elem_type)

def get_beamline_length(start, end):
    """A copy of `BeamLine.get_beamline_length(start,end)`
    """
    return BeamLine.get_beamline_length(start, end)

def get_next_element(elem_name):
    """A copy of `BeamLine.get_next_element(start,end)`
    """
    return BeamLine.get_next_element(elem_name)

def get_twiss_mismatch(twiss1, twiss2):
    """Returns the MisMatch Factor between to sets of Twiss parameters
    where Twiss is (alpha, beta, eps)

    Arguments:
       twiss1(list of doubles): [alpha, beta, emittance]
       twiss2(list of doubles): [alpha, beta, emittance]
    """
    a1, b1, e1 = twiss1
    g1 = (1.0 + a1 * a1) / b1
    a2, b2, e2 = twiss2
    g2 = (1.0 + a2 * a2) / b2
    r = b1 * g2 + g1 * b2 - 2.0 * a1 * a2
    mmf = math.sqrt(0.5 * (r + math.sqrt(r * r - 4))) - 1.0
    return mmf

def betalambda(mass, freq, w):
    """Return value of beta*lambda of beam.

    Arguments:
       mass(double): mc^2 of beam particle in MeV
       freq(double): frequency in MHz
       w(double): Kinetic energy in MeV
    """
    c = 2.99792458e8 # m/s
    wavelength = c / (freq * 1.0e6)
    gamma = 1.0 + w / mass
    beta = math.sqrt(1.0 - 1/(gamma * gamma))
    return beta * wavelength
    
