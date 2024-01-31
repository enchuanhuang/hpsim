
"""
SpaceCharge class
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""
import logging

import HPSim

class SpaceCharge():
    """An hpsim class for defining and modifying the space charge used in 
    the simulation"""
    spacecharge = ""

    def __init__(self, nr, nz, interval, adj_bunch, type="'scheff'"):
        """Creates an instance of the space-charge class
        
        Arguments:
           nr (int): number of space-charge mesh slices in r direction
           nz (int): number of space-charge mesh slices in z direction
           interval (double): maximum spacing between space charge kicks
           adj_bunch (int): number of adj_bunch used in s.c. calc
           type (str, optional): "scheff" by default and is the only option 
                                  at the moment
        """
        request = "HPSim.SpaceCharge("
        request += "nr = " + str(nr)
        request += ", nz = " + str(nz)
        request += ", interval = " + str(interval) #in meters, not user units
        request += ", adj_bunch = " + str(adj_bunch)
        #request += ", type = " + type
        request += ")"
        #print request
        self.spacecharge = eval(request)
        SpaceCharge.spacecharge = self.spacecharge

    def get_interval(self):
        """Returns the interval, i.e. maximum drift distance (m) between 
        space-charge kicks"""
        return self.spacecharge.get_interval()
    
    def set_interval(self, interval):
        """Set maximum drift distance (m) between space-charge kick

        Argument:
           interval (double): maximum distance between space-charge kicks
        """
        self.spacecharge.set_interval(interval)

    def get_adj_bunch(self):
        """Return the number of adjacent bunches in space charge calculation"""
        return int(self.spacecharge.get_adj_bunch())

    def set_adj_bunch(self, adj_bunch):
        """Set the number of adjacent bunches used in space charge calculation
        Argument:
           adj_bunch (int): number of adjacent bunches to use in s.c. calc.
        """
        self.spacecharge.set_adj_bunch(int(adj_bunch))

    def get_adj_bunch_cutoff_w(self):
        """Return the cutoff energy (MeV) above which the adjacent bunches are
        no longer used in space charge calculation and s.c. mesh region based 
        upon beam size, i.e. 3*sigmas. This enables automatic transition to 
        faster s.c. calc once adjacent bunches need no longer be considered"""
        return self.spacecharge.get_adj_bunch_cutoff_w()

    def set_adj_bunch_cutoff_w(self, w_cutoff):
        """Set the cutoff energy (MeV) above which the adjacent bunchss are
        no longer used in space charge calculation and s.c. mesh region based 
        upon beam size, i.e. 3*sigmas. This enables automatic transition to 
        faster s.c. calc once adjacent bunches need no longer be considered

        Argument:
           w_cutoff (double): threshold energy above which adjacent bunches are no
                              longer used in s.c. calc
        """

        self.spacecharge.set_adj_bunch_cutoff_w(w_cutoff)

    def get_mesh_size(self):
        """Return a list of floats representing the r,z mesh size"""
        return self.spacecharge.get_mesh_size()

    def set_mesh_size(self, nr, nz):
        """Set the size of the mesh, i.e. nr, nz
        
        Arguments:
           nr (double): number of radial grid points
           nz (double): number of longitudinal grid points
        """
        self.spacecharge.set_mesh_size(nr, nz)

    def get_mesh_size_cutoff_w(self):
        """Return the cutoff energy for the beam at which the mesh size will
        decrease by nr/2 and nz/2 and interval increase by 4.This enables 
        automatic transition to faster s.c. calc."""
        return self.spacecharge.get_mesh_size_cutoff_w()

    def set_mesh_size_cutoff_w(self, w_cutoff):
        """Set the cutoff energy for decreasing the mesh by nr/2, nz/2 and 
        increasing interval by 4. This enables automatic transition to 
        faster s.c. calc.
        
        Arguments:
           w_cutoff (double): Threshold energy (MeV) where s.c. calc reduces 
                              nr by factor 2, nz by factor 2 and interval by factor 4.
        """
        self.spacecharge.set_mesh_size_cutoff_w(w_cutoff)

    def get_remesh_threshold(self):
        """Get the remeshing factor (default is 0.05) where
        0 => remesh before every space-charge kick
        >0 => adaptive algorithm determines how much beam shape can change 
        before mesh must be redone"""
        return self.spacecharge.get_remesh_threshold()

    def set_remesh_threshold(self, rm_thres):
        """Set the remeshing factor (default is 0.05) where
        0 => remesh at before every space-charge kick
        >0 => adaptive algorithm determines how much beam shape can change 
        before mesh must be redone
        
        Arguments:
           rm_thres (double): the factor that determines if the s.c. grid is remeshed
                              or not.
        """

        self.spacecharge.set_remesh_threshold(rm_thres)