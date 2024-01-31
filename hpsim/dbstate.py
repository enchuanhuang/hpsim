"""
DBState class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""

from .dbconnection import DBConnection
from .helper_functions import *
class DBState():
    """An hpsim class capturing the state of all EPICS PV's in the connected dB's"""

    def __init__(self):
        """Create dB state object"""
        self.db_array=[]

    def get_db_pvs(self, file = None):
        """Record all Epics PV names and values from db 
        If filename present then also write to that file

        Arguments:
           file (str, optional): filename to write output to

        """
        self.epics_pvs = DBConnection.dbconnection.get_epics_channels()
        db_array = []
        for pv in self.epics_pvs:
            val = get_db_epics(pv)
            db_array.append([pv, val])
        self.db_array = db_array
        print("*** dB PV's stored in state object ***")
        if file is not None:
            # store db_pvs in file
            fid = open(file,'w')
            fid.write(self.db_array)
            fid.close()
            print("*** dB PV's written to file: ", file, " ***")
            
    def restore_db_pvs(self, file = None):
        """Restore EPICS PV in file or DBState object back into dB
        If file present use file else use DBState object

        Arguments:
           file (str, optional): filename from which to extract dB Epics PV values

        """
        if file is not None:
            # restore from file
            fid = open(file,'r')
            loc_db_array = fid.read()
            fid.close()
            for item in self.db_array:
                pv, val = item
                set_db_epics(pv, val)
                print("*** dB PV's restored from file: ", file, " ***")
        else:
            # restore from db_array object
            for item in self.db_array:
                pv, val = item
                set_db_epics(pv, val)
                print("*** dB PV's restored from db_array object ***")

    def print_pvs(self, pvname = None):
        """Print vals of EPICS PVs in DBState object that correspond to pvname
        Print all PVs vals in state object if pvname is not supplied 
        
        Arguments:
           pvname (str, optional): print value of named Epics PV

        """
        print('*** PV vals in state object ***')
        if pvname is not None:
            loc_pv = lcs.expand_pv(pvname)
        for item in self.db_array:
            pv, val = item
            if pvname is None:
                print('{0} = {1}'.format(pv, val))
            elif pv[0:len(loc_pv)] == loc_pv:
                print('{0} = {1}'.format(pv, val))

    def turn_off(self, pv_name):
        """Set all PV's with name pv_name to val of zero
        
        Arguments:
           pv_name(str): name of Epics PV

        """
        loc_pv = lcs.expand_pv(pv_name)
        for item in self.db_array:
            pv, val = item
            if pv[0:len(loc_pv)] == loc_pv:
                set_db_epics(pv, 0.0)
        
    def turn_on(self, pv_name):
        """Restore all PV's with name to associated vals from DBState
        
        Arguments:
           pv_name(str): name of Epics PV
        """
        loc_pv = lcs.expand_pv(pv_name)
        for item in self.db_array:
            pv, val = item
            if pv[0:len(loc_pv)] == loc_pv:
                set_db_epics(pv, val)

