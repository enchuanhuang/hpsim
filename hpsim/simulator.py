
"""
Simulator class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""

import HPSim
from .beam import Beam
from .beamline import BeamLine
from .spacecharge import SpaceCharge
from .plotdata import PlotData

class Simulator():
    """An hpsim class for defining the simulator"""

    # functions from original HPSim.so 
    def __init__(self, beam):
        """Creates an instance of the simulator class

        Arguments:
           beam (object): beam class object

        Returns:
           Simulator class object
        """
        request = "HPSim.Simulator(beam.beam, BeamLine.beamline, "
        request += "SpaceCharge.spacecharge,"
        request += "PlotData._plotdata)"
        self.sim = eval(request)

    def simulate(self, start_elem_name, end_elem_name):
        """Simulate from 'start' element to 'end' element, inclusive"""
        self.sim.simulate(start_elem_name, end_elem_name)

    def set_space_charge(self, state='off'):
        """Turn space charge on or off
        state (str, optional): "on", "off"(default)
        """
        self.sim.set_space_charge(state)