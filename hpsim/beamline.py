"""
BeamLine class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""
from typing import List
import logging
import pandas as pd
import numpy as np

import HPSim
from .dbconnection import DBConnection
from .constants import *

class BeamLine():
    """An hpsim class for defining and accessing the beamline
    """
    beamline = ""
    def __init__(self):
        """
        Arguments: none
        Returns:
           beamline object
        """
        # create class object
        BeamLine.beamline = HPSim.BeamLine(DBConnection.dbconnection)

    # functions from original HPSim.so 
    @classmethod
    def print_out(cls):
        """Print complete beamline listing from pinned memory, element by element
        for benchmarking"""
        cls.beamline.print_out()

    @classmethod
    def print_range(cls, start_element: str, end_element: str):
        """Print range of elements in pinned memory from start to end,
        for benchmarking

        Arguments:
           start_element (str): first element name in range to print
           last_element (str): last element name in range to print

        """
        cls.beamline.print_range(start_element, end_element)

    @classmethod
    def get_element_names(cls, start_element="", end_element="", elem_type=""):
        """Get list of elements in beamline from start_element to end_element
        with option elem_type
        
        Arguments:
           start_element(str): first element to retrieve from beamline
           end_element(str): last element to retrieve from beamline
           elem_type(str, optional): type of element, e.g., Diagnostics, ApertureC, ApertureR,
           Buncher, Dipole, Quad, Drift, RFGap-DTL, RFGap-CCL, SpchComp, Rotation, Steerer

        Return:
           Python list of element names

        """
        return cls.beamline.get_element_names(start_element, end_element, elem_type)

    @classmethod
    def get_element_model_indices(cls, start_element="", end_element="", elem_type=""):
        """Get model indices of elements in beamline from start_element to end_element
        with option elem_type
        
        Arguments:
           start_element(str): first element to retrieve from beamline
           end_element(str): last element to retrieve from beamline
           elem_type(str, optional): type of element, e.g., Diagnostics, ApertureC, ApertureR,
           Buncher, Dipole, Quad, Drift, RFGap-DTL, RFGap-CCL, SpchComp, Rotation, Steerer

        Return:
           Python list of element names

        """
        return cls.beamline.get_element_model_indices(start_element, end_element, elem_type)

    @classmethod
    def get_element_name(cls, index: int):
        """Get element name at certain index
        
        Arguments:
           index(int): index of the element name to get 

        Return:
           Element name
        """
        #BeamLine.beamline.get_element_names()#start_element, end_element, elem_type)
        return cls.beamline.get_element_name(index)

    # ----- monitor ------
    @classmethod
    def is_monitor_on(cls, element: str):
        """Check whether monitor is on for an element

        Arguments:
        element(str): element to check

        Return:
        status(bool): whether monitor is on or off
        """
        return cls.beamline.is_monitor_on(element)

    @classmethod
    def set_monitor_on(cls, element: str):
        """set monitor on for an element.

        Arguments:
        element(str): element to turn on monitor

        Return:
        status(bool): whether monitor is on or off
        """
        return cls.beamline.set_monitor_on(element)

    @classmethod
    def set_monitor_off(cls, element: str):
        """set monitor off for an element

        Arguments:
        element(str): element to turn off monitor

        Return:
        status(bool): whether monitor is on or off
        """
        return cls.beamline.set_monitor_off(element)

    def get_monitor_numbers(cls, start_element: str="", end_element: str= ""):
        """Get number of diagnostics elements that have field monitor = True in the database 
        from start_element to end_element

        Arguments:
           start_element(str): first element to consider from beamline
           end_element(str): last element to consider from beamline

        Return:
           number of diagnostics elements that are being monitored
        """
        return cls.beamline.get_num_of_monitors(start_element, end_element)


    # relocate functions
    @classmethod
    def get_element_list(cls, start_elem_name = "", end_elem_name = "", 
                         elem_type = ""):
        """Retrieve a list containing the names of beamline elements from 
        'start_elem_name' to 'end_elem_name'

        Arguments:
        start_elem_name(str): first element in list
        end_elem_name(str): last element in list
        elem_type(str): type of element (db type or C++ type) to retrieve

        """
        elem_type_dict = {'caperture':'ApertureC', 'raperture':'ApertureR',\
                            'buncher':'Buncher', 'diagnostics':'Diagnostics',\
                            'dipole':'Dipole', 'drift':'Drift', 'quad':'Quad',\
                            'dtl-gap':'RFGap-DTL', 'ccl-gap':'RFGap-CCL',\
                            'rotation':'Rotation', 'spch_comp':'SpchComp'}

        if elem_type in list(elem_type_dict.keys()):
            elem_type_resolved = elem_type_dict[elem_type]
        else:
            elem_type_resolved = elem_type

        ret = cls.get_element_names(start_elem_name, end_elem_name, 
                                    elem_type_resolved)
        return ret

    ''' depreciated. too slow to access database
    @classmethod
    def get_element_length(cls, elem_name: str):
        """Return length of beamline element in hpsim base units(m).
        
        Arguments:
        elem_name(str): name of element

        """
        elem_type = get_db_model(elem_name, 'model_type')
        eff_len = 0.0
        if elem_type in ['drift', 'quad', 'dtl_gap', 'ccl_gap']:
            # elements with a defined length
            eff_len = get_db_model(elem_name, 'length_model')
        elif elem_type in ['dipole']:
            # effective path length must be calculated
            eff_len = get_db_model(elem_name, 'rho_model') \
                * get_db_model(elem_name, 'angle_model')
        return eff_len
    '''

    @classmethod
    def get_element_length(cls, elem_name: str):
        """Return length of beamline element in hpsim base units(m) using C++ function
        
        Arguments:
        elem_name(str): name of element
        """
        return cls.beamline.get_beam_travel_length_element(elem_name)
    
    @classmethod
    def get_element_lengths(cls, start_elem: str, stop_elem: str):
        """Return length of beamline element in hpsim base units(m) using C++ function
        
        Arguments:
        elem_name(str): name of element
        """
        return cls.beamline.get_beam_travel_length_elements(start_elem,
                                                            stop_elem)

    @classmethod
    def get_beamline_direction(cls, start_elem: str, stop_elem: str):
        """Returns +1 for stop_elem beyond start_elem or -1 if stop_elem behind start_elem
        
        Arguments:
        start_elem(str): beginning element name
        stop_elem(str): final element name
        """
        direction = 1
        beamline = cls.get_element_list()
        istart = beamline.index(start_elem)
        istop = beamline.index(stop_elem)
        if istart > istop:
            direction = -1
        return direction    

    ''' depreciated. too low accessing db
    @classmethod
    def get_beamline_length(cls,start: str, end: str):
        """Returns length of beamline from element 'start' to element 'end'
        in hpsim base units (m). If start is after stop, then the length is < 0

        Arguments:
        start(str): first element in list
        end(str): last element in list

        """
        l = 0.0
        direction = cls.get_beamline_direction(start, end)
        if direction < 0:
            loc_start, loc_end = end, start
        else:
            loc_start, loc_end = start, end
        bl = cls.get_element_list(start_elem_name = loc_start, end_elem_name = loc_end)
        for elem in bl:
            print(elem)
            #print elem, get_element_length(elem)
            l += cls.get_element_length(elem)
        return l * direction
    '''

    @classmethod
    def get_beamline_length(cls,start: str="", end: str=""):
        """Returns length of beamline from element 'start' to element 'end'
        in hpsim base units (m). If start is after stop, then the length is < 0

        Arguments:
        start(str): first element in list
        end(str): last element in list

        """
        l = 0.0
        direction = cls.get_beamline_direction(start, end)
        if direction < 0:
            loc_start, loc_end = end, start
        else:
            loc_start, loc_end = start, end

        return direction*cls.beamline.get_beam_travel_length_range(loc_start, loc_end)

    @property
    def data(self):
        df = pd.DataFrame(columns=["name", "type", "model_index",
                                   "z_start", "z_mid", "z_end", "length", ],
                                   dtype=float)
        df["name"] = self.beamline.get_element_names()
        df["type"]  = self.beamline.get_element_types()
        df["length"] = self.beamline.get_beam_travel_length_elements() * USER_UNITS["z"]
        df["z_end"] = df["length"].cumsum()
        df["z_start"] = df["z_end"] - df["length"]
        df["z_mid"] = (df["z_start"] + df["z_end"])/2.
        df["model_index"]  = self.beamline.get_element_model_indices()
        df.set_index("model_index", drop=True, inplace=True)

        # get monitored info
        idx_monitored = self.beamline.get_monitored_indices("", "")
        df["monitor"] = False
        df.loc[idx_monitored, "monitor"] = True
        return df.copy()

    @classmethod
    def get_midpoints(cls, start: str="", end: str=""):
        """Returns a list of the distance to the midpoint of each element in 
        the complete beamline, units (m). 

        Arguments: None

        Return:
        element_names(list) : list of element names
        midpoints(list)     : list of midpoints
        """
        df = cls.get_beamline_info()
        idx1, idx2 = df.index[0], df.index[-1]
        if start!="":
            if start in df["name"]:
                idx1 = df.index[df["name"] == start]
            else:
                logging.error(f"Cannot find {start}")
                return None
        if end!="":
            if end in df["name"]:
                idx2 = df.index[df["name"] == end]
            else:
                logging.error(f"Cannot find {end}")
                return None
        return df.loc[idx1:idx2, "name"], df.loc[idx1:idx2, "z_mid"]

    @classmethod
    def get_midpoints_by_names(cls, names: List[str]):
        """Returns a list of the distance to the midpoint of each element in 
        the range (m). 

        Arguments:
        names(List[str]) : list of element names

        Return:
        element_names(list) : list of element names
        midpoints(list)     : list of midpoints
        """
        names = list(names)
        df = cls.get_beamline_info()
        df = df.drop_duplicates("name").set_index("name")
        # make sure all names are in list
        wrong_names = set(names) - set(df.index)
        if len(wrong_names)!=0:
            for wrong_name in wrong_names:
                names.pop(wrong_name)
                logging.error(f"{wrong_name} not in beamline")
        print(df.loc[names, "z_mid"])
        return names, list(df.loc[names, "z_mid"])

    @classmethod
    def get_first_element(cls):
        """Returns name of first element in connected database

        Arguments: None
        """
        return cls.get_element_list()[0]

    @classmethod
    def get_last_element(cls):
        """Returns name of first element in connected database

        Arguments: None
        """
        return cls.get_element_list()[-1]

    @classmethod
    def get_next_element(cls, elem_name: str):
        """Returns the name of the next element in the connected databases
        Arguments:
        elem_name(str): name of element
        """
        beamline = cls.get_element_list()
        if elem_name in beamline:
            next_idx = beamline.index(elem_name) + 1
            if next_idx < len(beamline):
                return beamline[next_idx]
            else:
                logging.warning(f"{elem_name} is the last element")
                return None
        else:
            logging.warning(elem_name, "not found in beamline list")
            return None

    @classmethod
    def get_positions_by_indices(cls, zs:np.ndarray, idxs:np.ndarray):
        """Returns the positions in zs by input index array idxs
        
        Arguments:
        zs(np.ndarray): returns value
        idxs (np.ndaray)  : index to get
        """
        # assign mid points and exclude too large ones
        nelm = len(zs)
        vals = np.zeros(len(idxs), dtype=float)
        m = (idxs>=0) & (idxs<nelm) # valid ones
        mF = ~m
        vals[m] = zs[idxs[m]] # assign by index
        vals[mF] = -99999
        idxsF = set(idxs[mF])
        for idx in idxsF:
            nF = np.sum(idxs==idx)
            logging.warning(f"{nF} of index {idx} does not exist. -99999 assigned")
        return vals
