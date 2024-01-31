import sys
import os
import logging
import time
import yaml
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# HPSIm related
import hpsim as hps
from hpsim import lcsutil as lcs
from hpsim import sqldb as pydb
from hpsim import LossDetectors
from hpsim import nputil as npu
from hpsim import get_twiss_mismatch


def create_hpsim_model(config: dict):
    """
    Create a hpsim model with a config dictionary. The dictionary should 
    contain sections including: basic, beam, simulation, database, and monitor.
    variables in those sections will be used to initialize the hpsim model. An
    example of the dictionary looks like below (configured in yaml):
    ```yaml
        basic:
            GPU : 0
            log_level : 2 # 0: CRITICAL, ERROR, WARNING, INFO, DEBUG
            debug_level : 1 # used for C++.


        beam:
            num : 65536  # number of particles
            mass : 938.272 # proton mass
            charge : 1.0   # H+
            current : 0.006 # [A]
            alpha_x : 0.492 # 
            beta_x  : 201   # caluate under cm & rad. See tutorial/LEBT_beam_initialization
            emittance_x : 0.0004065 # 6 rms
            alpha_y : -0.441
            beta_y : 115
            emittance_y : 0.000404
            delta_phi : 180  # [deg]
            synch_phi : 0    # [deg]
            synch_w   : 0.750 # [MeV]
            random_seed : 0

        simulation:
            SIM_START : "TADB01"
            SIM_STOP : "IPFTGT"

        database:
            db_dir : "./db/"
            tmpdir : "./db/tmp"
            dbs : ['tatd-20221022.db', "dtl-20221022hp.db", "tripf.db"]

        # devices where beam info is saved.
        monitor:
            devices : ["QL", "QM","QD","EM", "AB","CM", "DT", "PM", "WS"]
    """
    model = HPSimModel(**(config["basic"]))
    model.set_sim_range(**(config["simulation"]))
    model.set_beam(**(config["beam"]))
    model.set_db(**(config["database"]))
    model.set_monitor(**(config["monitor"]))
    model.initialize()
    return model


class HPSimModel(object):
    par_dir = Path(__file__).parent.parent
    def __init__(self, GPU=0,
                      log_level=3, debug_level=1, **kwargs):
        self.GPU = GPU
        hps.set_debug_level(debug_level)
        hps.set_log_level(log_level)
        self._epics_values = dict()   
        self.monitor_devices = []

    def set_db(self, db_dir, dbs, tmpdir=None, **kwargs):
        self.db_dir = Path(db_dir)
        self.dbs = dbs

        if tmpdir is not None:
            self.tmpdir = Path(tmpdir)
        else:
            self.tmpdir = self.db_dir / "tmp"

    def set_sim_range(self, SIM_START="TBDB02", SIM_STOP = "TREM02", **kwargs):
        self.SIM_START = SIM_START
        self.SIM_STOP = SIM_STOP

    def set_beam(self, num=1024*128, current=0.015, 
                      sync_w = 0.750, DELTA_PHI=0, mass=939.294, charge = -1, 
                      ax= -1.2, bx= 130, ex= 0.003, 
                      ay= -1.2, by= 130, ey= 0.003, **kwargs):
        self.num = num
        self.current = current
        self.sync_w = sync_w
        self.DELTA_PHI = DELTA_PHI
        self.mass = mass
        self.charge = charge

        self.ax = ax
        self.bx = bx
        self.ex = ex
        self.ay = ay
        self.by = by
        self.ey = ey
         

    def set_monitor(self, devices, **kwargs):
        self.monitor_devices = devices
    
    def initialize(self, **kwargs):
        # connect to database
        db_dir = self.db_dir
        dbs = self.dbs

        lib_dir = self.par_dir / 'db/lib'
        print(lib_dir)
        dbconn = hps.DBConnection(db_dir, dbs, lib_dir, 'libsqliteext.so', self.tmpdir)
        dbconn.print_dbs()
        logging.info("dB connection Initialized")

        # create beamline
        bl = hps.BeamLine()
        beamline = hps.get_element_list()
        logging.info("Beamline Initialized")

        # create H- beam
        beam = hps.Beam(mass=self.mass, charge=self.charge, 
                        current=self.current, num=self.num) #H- beam
        beam.set_dc(self.ax, self.bx, self.ex, self.ay, self.by, self.ey, 
                    180.0, 0.0, self.sync_w) #TBDB02 20140901
        beam.set_frequency(201.25)
        betalambda = hps.betalambda(mass = beam.get_mass(), 
                                freq=beam.get_frequency(), w= self.sync_w)
        phi_offset = -hps.get_beamline_length(self.SIM_START,'BLZ')/betalambda *360 + self.DELTA_PHI
        beam.set_ref_w(self.sync_w)
        beam.set_ref_phi(phi_offset)
        beam.translate('phi', phi_offset)
        beam.save_initial_beam()
        logging.info("H- beam Initialized")

        # create spacecharge
        spch = hps.SpaceCharge(nr = 32, nz = 128, interval = 0.025, adj_bunch = 3)
        spch.set_adj_bunch_cutoff_w(0.8)
        spch.set_remesh_threshold(0.02)

        # create plot data (monitor)
        # add more monitors
        # ----------- Setting up elements for monitor -------------
        for element in beamline:
            if element[2:4] in self.monitor_devices:
                ret = bl.set_monitor_on(element)
                if ret is False:
                    print(element)
        logging.info(f"Number of monitors: {bl.get_monitor_numbers()}")
        nmonitor = bl.get_monitor_numbers()
        plotdata = hps.PlotData(nmonitor)

        # create simulator
        sim = hps.Simulator(beam)
        sim.set_space_charge('on')
        logging.info("Simulator Initialized")

        ## retract jaw
        #hps.set_db_model('TAFJ03', 'in_out_model', 0)
        
        # initialize variables
        # log level
        self._beam = beam
        self._sim = sim
        self._spch = spch
        self._plotdata = plotdata
        self._bl = bl
        self._dbconn = dbconn
        self._epics_channels = dbconn.get_epics_channels()
        self.update_epics_values()

    def simulate(self):
        self._beam.restore_initial_beam()
        self._plotdata.reset()
        self._dfbl = self._bl.data
        logging.debug(f"Simulate from {self.SIM_START} to {self.SIM_STOP}")
        self._sim.simulate(self.SIM_START, self.SIM_STOP)
        self._dfbl = self._plotdata.update_beamline_dataframe(self._dfbl)

        # obtain data
        self._dfbeam = self._beam.data
        self._dfbl = self._bl.data
        self._dfbl = self._plotdata.update_beamline_dataframe(self._dfbl)


    @property
    def cost(self):
        cost = self.mismatch(-0.0575, 5.95, 0.0271, 26.01) # magic number at BLZ
        return cost
    
    def mismatch(self, ax0, bx0, ay0, by0):
        # calculate mismatch between the beam twiss parameter and the input
        
        mask =  self.beam.get_good_mask()
        ax, bx, ex = self.beam.get_twiss("x", mask)
        ay, by, ey = self.beam.get_twiss("y", mask)
        bx0 = bx/1000.
        by0 = by/1000.
        bx = bx/1000.
        by = by/1000.
        logging.debug(f"Getting mismatch between "
                     f"({ax:.4f}, {bx:.4f}, {ay:.4f}, {by:.4f}) and"
                     f"({ax0:.4f}, {bx0:.4f}, {ay0:.4f}, {by0:.4f})"
                     )
        M = 0
        delx = (np.power(bx0-bx, 2) + np.power(ax*bx0-ax0*bx,2))/(bx0 * bx)
        M += 0.5*(delx + np.sqrt(delx*delx + 4*delx ));
        dely = (np.power(by0-by, 2) + np.power(ay*by0-ay0*by,2))/(by0 * by)
        M += 0.5*(dely + np.sqrt(dely*dely + 4*dely ));
        return M, ax, bx*1e3, ay, by*1e3

 

        return 1 - self.transmission 
    
    def update_epics_values(self):
        """get current pvval from database for all variables"""
        for pvname in self._epics_channels:
            self._epics_values[pvname] = hps.get_db_epics(pvname)

    def evaluate(self, vals: dict) -> float:
        """ Xopt style of evaulate. Input variables that need to change and
        return cost.
        """
        # update
        logging.debug("input values" + ",".join(vals))
        for pvname, val in vals.items():
            if pvname not in self._epics_channels:
                logging.error(f"{pvname} not available. Ignored.")
            hps.set_db_epics(pvname, val)
            self._epics_values[pvname] = val
        #simulate
        self.simulate()
        dfbl = self.dfbl
        xavg_max = np.abs(dfbl["xavg"]).max()
        yavg_max = np.abs(dfbl["yavg"]).max()
        xavg_std = np.abs(dfbl["xavg"]).std()
        yavg_std = np.abs(dfbl["yavg"]).std()
        xpavg_max = np.abs(dfbl["xpavg"]).max()
        ypavg_max = np.abs(dfbl["ypavg"]).max()
        xpavg_std = np.abs(dfbl["xpavg"]).std()
        ypavg_std = np.abs(dfbl["ypavg"]).std()
        mismatch, ax, bx, ay, by = self.mismatch(-0.0575, 5.95, 0.0271, 26.01) # magic number at BLZ
        xsig = self._beam.get_sig("x")
        ysig = self._beam.get_sig("y")
        sig_ratio_diff = np.abs(xsig/ysig-0.5)
        total_sig = np.sqrt(xsig**2 + ysig**2)
        cost = total_sig*30+ sig_ratio_diff*5
        ret = {"mismatch": mismatch,
               "cost": cost,
               "ax": ax,
               "ay": ay,
               "bx": bx,
               "by": by,
               "sig_ratio_diff": sig_ratio_diff,
               "xsig": xsig,
               "ysig": ysig,
               "total_sig": total_sig,
               "dfbl": self.dfbl,
               "time": time.time(),
               'xavg_max': xavg_max,
               'yavg_max': yavg_max,
               'xavg_std': xavg_std,
               'yavg_std': yavg_std,
               'xpavg_max': xpavg_max,
               'ypavg_max': ypavg_max,
               'xpavg_std': xpavg_std,
               'ypavg_std': ypavg_std,
            }
        logging.debug(f"(mismatch) = ("
                    f"{ret['mismatch']:.4f}")
        # get cost
        return ret
    
    

    def plot_standard(self):
        mask = self._beam.get_good_mask()
        hps.BeamPlot.create_standard_plots(self._beam, mask)

    @property
    def beam(self):
        return self._beam
    
    @property
    def bl(self):
        return self._bl

    @property
    def plotdata(self):
        return self._plotdata

    @property
    def dbconn(self):
        return self._dbconn
    
    @property
    def transmission(self):
        nalive = np.sum(self._dfbeam.lost_idx==0)
        npar = len(self._dfbeam)
        return nalive / npar

    
    @property
    def dfbeam(self):
        dfbeam = self._dfbeam.copy()
        return dfbeam


    @property
    def dfbl(self):
        dfbl = self._dfbl.copy()
        last_idx = dfbl[dfbl.name==self.SIM_STOP].index[0]
        dfbl = dfbl.loc[:last_idx].copy()
        return dfbl
    
    @property
    def pvnames(self)->List[str]:
        # return list of pvnames in control
        return self._epics_channels.copy()

    @property
    def epics_channels(self)->List[str]:
        return self._epics_channels.copy()


    @property
    def epics_values(self)-> dict:
        return self._epics_values.copy()





if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(levelname)s:%(message)s')
    np.random.seed(310)
    dfoff = pd.DataFrame(columns=["name",  "dx", "dxp", "dy", "dyp"])
    dfoff["name"] = ["TBDR01", "TBDR05"]#, "TBDR09", "TBDR13", "TBDR17"]
                     #"TBDR21", "TBDR25", "TBDR29", "TBDR33", "TBDR37",
                     #"TBDR41", "TBDR45", "TBDR49", 
                     #"TDDR01", "TDDR05", "TDDR09", "TDDR13"]
    dfoff["dx"] = np.random.normal(0, 0.0, len(dfoff))
    dfoff["dy"] = np.random.normal(0, 0.0, len(dfoff))
    dfoff["dxp"] = 0
    dfoff["dyp"] = 0

    dfoff2 = pd.DataFrame(columns=["name",  "dx", "dxp", "dy", "dyp"])
    dfoff2["name"] = ["TBDR01", "TBDR05"]
    dfoff2.loc[0, "dx"] =  0.35
    dfoff2.loc[1, "dy"] = 0.0
    dfoff2.loc[0, "dxp"] = 0# -3.5
    dfoff2.loc[1, "dyp"] = 0.0

    model = HPSimModel(DELTA_PHI=90, SIM_STOP="TREM02", GPU=3, offsets=dfoff2, log_level=3, debug_level=1)
    model.initialize()
    #model.simulate()
    cost = model.evaluate({"TBSM101P01":0, "TBSM201P01":0.})
    cost = model.evaluate({"TBSM101P01":0, "TBSM201P01":0.})
    cost = model.evaluate({"TBSM101P01":0, "TBSM201P01":0.})
    cost = cost["cost"]
    dfbl = model.dfbl
    mask = model.beam.get_good_mask()
    model.beam.print_results(mask)
    print(f"Transmission = {(1-cost)*100:.2f}")
    #model.plot_standard()
    #plt.show()
    #sys.exit()

    fig, ax = plt.subplots()

    m = dfbl["monitor"]
    print(dfbl[m][["name", "xavg"]])

    ax.plot(dfbl[m]["z_mid"]/100, dfbl["xavg"][m]*10, ".-")
    ax.set_xlabel("[m]")
    ax.set_ylabel("[mm]")
    #m2 = dfbl.name.str.contains("DR")
    #ax.plot(dfbl[m2]["z_mid"], dfbl["xavg"][m2], ".")
    for idx, row in dfbl[dfbl.type=="Dipole"].iterrows():
        ax.axvline(x=row["z_mid"]/100)
    ax.set_title(f"transmission = {model.transmission*100:.2f}%")

    ax2 = ax.twinx()
    ax2.set_ylabel('transmission', color="C1")  # we already handled the x-label with ax1
    ax2.plot(dfbl[m]["z_mid"]/100, 1-dfbl[m]["loss_ratio"], color="C1")
    ax2.tick_params(axis='y', labelcolor="C1")
    print(np.max(dfbl["xavg"][m]))
    print(np.sum(m))
    plt.show()
    

