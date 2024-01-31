"""
class for Beam
separated from hpsim.py on 2023/03/14 by En-Chuan Huang
"""
import os
import logging
from typing import List


#import various packages
import numpy as np
import pandas as pd
import math

# HPSim related ones
import HPSim as HPSim
from .constants import *

class Beam():
    """An hpsim class for manipulating beams as Numpy arrays"""

    def __init__(self, **args): #mass, charge, current, file = None, num = 0):
        """Creates an instance of the beam class

        Args:
           keywords are required and lowercase
           where
           
           file (string): Input file name containing beam distribution

            or

           mass (double): mc^2 of particle mass (MeV)
           charge (double): q/|e|
           current (double): bunch peak current(Amps)
           num (int): number of macro particles

        Attributes:
           initial_current(double): Initial beam current (A)
           initial_frequency(double): Initial freq (MHz)
           initial_size(int): Initial number of macroparticles in beam distr

        Returns:
            Returns a beam class object

        Examples:
            Beam(file = filename) 
            
             or
            
            Beam(mass = particle_mass, charge = particle_charge, 
                           current = beam_current, num = number of particles)

        """
        if 'file' in list(args.keys()):
            # create beam based upon contents of file
            if os.path.isfile(args['file']):
                self.beam = HPSim.Beam(file = args['file'])
                self.initial_current = self.beam.get_current()
                self.initial_frequency = self.beam.get_frequency()
                self.initial_gd_size = self.beam.get_size() - self.beam.get_loss_num()
            else:
                print('Input beam file, "',args['file'],'" not found! Exiting')
                exit()
        else:
            # create beam from supplied arguments
            self.beam = HPSim.Beam(mass=float(args['mass']),
                                   charge=float(args['charge']), 
                                   current=float(args['current']),
                                   num=int(args['num']))
            self.initial_current = float(args['current'])
            self.initial_frequency = 0
            self.initial_gd_size = self.get_initial_size()
        return
     
    def get_distribution(self, option='all'):
        """Returns a list of Numpy vectors containing the beam coordinates in 
        user units x, xp, y, yp, phi, w, loss = beam.get_distribution()

        Argument:
           option (str): 'good', 'bad' or 'all'=default
        """
        loss_all = self.get_losses()
        if option == 'all':
            loss = loss_all
        elif option == 'bad':
            loss = np.where(loss_all > 0)[0]
        elif option == 'good':
            loss = np.where(loss_all == 0)[0]
        return self.get_x(option), \
            self.get_xp(option), \
            self.get_y(option), \
            self.get_yp(option), \
            self.get_phi(option), \
            self.get_w(option), \
            loss

    def set_distribution(self, x, xp, y, yp, phi, w, loss = None):
        """Creates beam distribution using vectors of coordinates (users units)
        Arguments:
           x (Numpy vector double): x coordinates cm
           xp (Numpy vector double): xp coordinates mr
           y (Numpy vector double): y coordinates cm
           yp (Numpy vector double): yp coordinates mr
           phi (Numpy vector double): phi coordinates deg
           w (Numpy vector double): w coordinates MeV
           loss (Numpy vector int, optional): loss coordinate or 0-> good 
        """
        # first zero out distribution if array will be only partially filled
        if len(x) < self.beam.get_size():
            zeros = self.beam.get_size()*[0.0]
            self.beam.set_distribution(zeros, zeros, zeros, zeros, zeros, zeros)
        if loss is not None:
            self.beam.set_distribution(list(x/CM), list(xp/MILLIRAD), 
                                       list(y/CM), list(yp/MILLIRAD),
                                       list(phi/DEG), list(w), list(loss))
        else:
            self.beam.set_distribution(list(x/CM), list(xp/MILLIRAD), 
                                       list(y/CM), list(yp/MILLIRAD),
                                       list(phi/DEG), list(w))            
        return
    
    def set_waterbag(self, alpha_x, beta_x, emittance_x,
                     alpha_y, beta_y, emittance_y,
                     alpha_z, beta_z, emittance_z,
                     synch_phi, synch_w, frequency, random_seed = 0):
        """ Creates a 6D waterbag using PARMILA input units
        Arguments:
           alpha_x (double): x-plane Twiss alpha parameter
           beta_x (double): x-plane Twiss beta parameter (cm/radian)
           emittance_x (double): x-plane total emittance (cm * radian)
           alpha_y (double): y-plane Twiss alpha parameter
           beta_y (double): y-plane Twiss beta parameter (cm/radian)
           emittance_y (double): y-plane total emittance (cm * radian)
           alpha_z (double): z-plane Twiss alpha parameter
           beta_z (double): z-plane Twiss beta parameter (deg/MeV)
           emittance_z (double): z-plane total emittance (deg * MeV)
           synch_phi (double): synchronous phase (deg)
           synch_w (double): synchronous energy (MeV)
           frequency (double): frequency (MHz)
           random_seed (option [int]): random seed for generating distribution
        """
        if random_seed:
            self.beam.set_waterbag(float(alpha_x), float(beta_x), float(emittance_x),
                             float(alpha_y), float(beta_y), float(emittance_y),
                             float(alpha_z), float(beta_z*DEG), float(emittance_z/DEG),
                             float(synch_phi/DEG), float(synch_w), float(frequency),
                             int(random_seed))
        else:
            self.beam.set_waterbag(float(alpha_x), float(beta_x), float(emittance_x),
                             float(alpha_y), float(beta_y), float(emittance_y),
                             float(alpha_z), float(beta_z*DEG), float(emittance_z/DEG),
                             float(synch_phi/DEG), float(synch_w), float(frequency))
        return
    
    def set_dc(self, alpha_x, beta_x, emittance_x,
               alpha_y, beta_y, emittance_y,
               delta_phi, synch_phi, synch_w, random_seed = 0):
        """ Creates DC beam using PARMILA input units set_waterbag
        Arguments:
           alpha_x (double): x-plane Twiss alpha parameter
           beta_x (double): x-plane Twiss beta parameter (cm/radian)
           emittance_x (double): x-plane total emittance (cm * radian)
           alpha_y (double): y-plane Twiss alpha parameter
           beta_y (double): y-plane Twiss beta parameter (cm/radian)
           emittance_y (double): y-plane total emittance (cm * radian)
           alpha_z (double): z-plane Twiss alpha parameter
           beta_z (double): z-plane Twiss beta parameter (deg/MeV)
           emittance_z (double): z-plane total emittance (deg * MeV)
           delta_phi (double): half-width of phase distribution (deg)
           synch_phi (double): synchronous phase (deg)
           synch_w (double): synchronous energy (MeV)
           random_seed (int, optional): random seed for generating distribution
        """
        if random_seed:
            self.beam.set_dc(float(alpha_x), float(beta_x), float(emittance_x),
                       float(alpha_y), float(beta_y), float(emittance_y),
                       float(delta_phi/DEG), float(synch_phi/DEG), float(synch_w),
                       int(random_seed))
        else:
            self.beam.set_dc(float(alpha_x), float(beta_x), float(emittance_x),
                       float(alpha_y), float(beta_y), float(emittance_y),
                       float(delta_phi/DEG), float(synch_phi/DEG), float(synch_w))
        return
    
    def save_initial_beam(self):
        """Save initial beam distribution for later restore."""
        self.beam.save_initial_beam()
        return

    def save_intermediate_beam(self):
        """Save intermediate beam distribution for later restore."""
        self.beam.save_intermediate_beam()
        return
    
    def restore_initial_beam(self):
        """Restore initial beam distribution for next simulation."""
        self.beam.restore_initial_beam()
        #PlotData.reset()
        return
    
    def restore_intermediate_beam(self):
        """Restore intermediate beam distribution for next simulation."""
        self.beam.restore_intermediate_beam()
        return
    
    def print_simple(self):
        """Print particle coordinates x, x', y, y', phi, w coordinates to screen"""
        self.beam.print_simple()
        return
    
    def print_to(self, output_file_name):
        """Print particle coordinates x, x', y, y', phi, w coordinates to file"""
        self.beam.print_to(output_file_name)
        return
    
    def set_ref_w(self, w):
        """Set reference particle energy, MeV"""
        self.beam.set_ref_w(float(w))
        return
    
    def get_ref_w(self):
        """Return the reference particle's energy in MeV"""
        return self.beam.get_ref_w()

    def set_ref_phi(self, phi):
        """Set reference particle phase, degrees"""
        self.beam.set_ref_phi(float(phi / DEG))
        return
    
    def get_ref_phi(self):
        """Return the reference particle's phase in degree"""
        return self.beam.get_ref_phi() * DEG

    def set_frequency(self, frequency):
        """Set beam frequency in MHz"""
        self.beam.set_frequency(float(frequency))
        if self.initial_frequency == 0:
            self.initial_frequency = self.get_frequency()
        return
    
    def get_frequency(self):
        """Return beam frequency in MHz"""
        return self.beam.get_frequency()

    def get_mass(self):
        """Return mass of beam, mc^2, in MeV"""
        return self.beam.get_mass()

    def get_charge(self):
        """Return charge of beam in q/|e|"""
        return self.beam.get_charge()

    def get_initial_size(self):
        """Returns initial number of beam macro particles"""
        return self.beam.get_size()

    def get_x(self, option = 'all'):
        """Return Numpy array of x coordinates (cm) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_x(option) * CM

    def get_xp(self, option = 'all'):
        """Return Numpy array xp coordinates (mr) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_xp(option) * MILLIRAD

    def get_y(self, option = 'all'):
        """Return Numpy array y coordinates of (cm) macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_y(option) * CM

    def get_yp(self, option = 'all'):
        """Return Numpy array yp coordinates of (mr) macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_yp(option) * MILLIRAD

    def get_phi(self, option = 'all', option2='absolute'):
        """Return Numpy array phi coordinates (deg) of macro particles, option = 'good', 
        'bad', 'all (default). option2 = 'relative', 'absolute' (default)"""
        return self.beam.get_phi(option, option2) * DEG



    def get_w(self, option = 'all'):
        """Return Numpy array w coordinates (MeV) of macro particles, option = 'good', 
        'bad', 'all (default)"""
        return self.beam.get_w(option)

    def get_losses(self):
        """Return Numpy array of loss condition of macro particles"""
        return self.beam.get_losses()

    def get_loss_num(self):
        """Return number of lost particles"""
        return self.beam.get_loss_num()

    def get_avg_x(self):
        """Return average x value of beam in cm"""
        return self.beam.get_avg_x() * CM

    def get_avg_y(self):
        """Return average y value of beam in cm"""
        return self.beam.get_avg_y() * CM

    def get_avg_phi(self, option = 'absolute'):
        """Return average phi value of beam in deg"""
        return self.beam.get_avg_phi(option) * DEG

    def get_avg_w(self):
        """Return average w value of beam in MeV"""
        return self.beam.get_avg_w()

    def get_sig_x(self):
        """Return sigma x of beam in cm"""
        return self.beam.get_sig_x() * CM

    def get_sig_y(self):
        """Return sigma y of beam in cm"""
        return self.beam.get_sig_y() * CM

    def get_sig_phi(self):
        """Return sigma phi of beam in deg"""
        return self.beam.get_sig_phi() * DEG

    def get_sig_w(self):
        """Return sigma w of beam"""
        return self.beam.get_sig_w()

    def get_emittance_x(self):
        """Return rms x emittance of beam cm*mr"""
        return self.beam.get_emittance_x() * CM * MILLIRAD

    def get_emittance_y(self):
        """Return rms y emittance of beam in cm*mr"""
        return self.beam.get_emittance_y() * CM * MILLIRAD

    def get_emittance_z(self):
        """Return rms z emittance of beam in Deg*MeV"""
        return self.beam.get_emittance_z() * DEG

    def apply_cut(self, axis, minval, maxval):
        """Remove particles from beam by apply cuts along 'x', 'y', 'p' or 'w'"""
        self.beam.apply_cut(axis, minval, maxval)
        return

    def translate(self, axis, value):
        """Translate particle coordinates along specified axis by given value"""
        # divide by to convert from user_units
        self.beam.translate(axis, value / USER_UNITS[axis.lower()])
        return
    
    @property
    def data(self) -> pd.DataFrame:
        """ Return a `pd.DataFrame` that contains (x, xp, y, yp, phi, w) and the 
        `lost_idx` (element model_index of which the particle is lost).
        """
        df = pd.DataFrame(columns=["x", "xp", "y", "yp", "phi", "w", "lost_idx"], dtype=float)
        df["x"] = self.get_x("all")
        df["y"] = self.get_y("all")
        df["w"] = self.get_w("all")    
        df["xp"] = self.get_xp("all")    
        df["yp"] = self.get_yp("all")    
        df["phi"] = self.get_phi("all", "absolute")
        df["phi_r"] = self.get_phi("all", "relative")
        df["lost_idx"] = self.get_losses()
        return df

    ################################ new functions #################################
    # These functions use create or employ masks that allow the user to be 
    # selective in what particles are returned or analyzed based upon the 
    # mask constraints

    def get_coor(self, var, mask = None):
        """Return vector of macro particle coordinates (in USER_UNITS) 
        after optional mask is applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        request = "self.beam.get_" + lc_var
        if lc_var in COORDINATES:
            request += "('all')"
            if mask is not None:
                coor = eval(request)[mask] * USER_UNITS[lc_var]
            else:
                coor = eval(request) * USER_UNITS[lc_var]
            if lc_var == 'phi':
                 coor=coor   
                #coor = modulo_phase(coor, self.get_ref_phi())
        elif lc_var in LOSSES:
            request += "()"            
            if mask is not None:
                coor = eval(request)[mask]
            else:
                coor = eval(request)
        else:
            print("Error: Empty array returned, variable", \
            str.upper(var), "not recognized")
            return np.array([])
        return coor

    def get_avg(self, var, mask = None):
        """Return average of beam coordinates after optional mask applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        if lc_var in COORDINATES:
            # if mask is an empty list then return average of 0.0 ljr v10
            #if mask == None:
            #    return np.mean(self.get_coor(lc_var, mask))
            #elif list(mask) == []:
            #    return 0.0
            #else:
            #    return np.mean(self.get_coor(lc_var, mask))
            if mask is not None:
                if list(mask) == []:
                    # if mask is an empty list then return average of 0.0
                    return 0.0
                else:
                    return np.mean(self.get_coor(lc_var, mask))
            else:
                return np.mean(self.get_coor(lc_var, mask))

        else:
            print("Error: Average not found. Variable", \
            str.upper(var), "not recognized")
            return float('nan')

    def get_sig(self, var, mask = None):
        """Return sigma of beam coordinates after optional mask applied.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        lc_var = var.lower()
        if var.lower() in COORDINATES:
            if lc_var == 'phi':
                #pcoor = self.get_coor(lc_var, mask)
                #ptmp = np.mod(pcoor, 360)
                #hval, hbin = np.histogram(ptmp, bins=360)
                #print self.beam.get_sig_phi(), np.std(pcoor), np.std(pcoor - hbin[np.argmax(hval)] + 180#)
                #return np.std(pcoor - hbin[np.argmax(hval)] + 180)
                #return np.std(pcoor - self.get_avg('phi', mask) + 180)
                return np.std(self.get_coor(lc_var, mask))
            else:
                return np.std(self.get_coor(lc_var, mask))
        else:
            print("Error: Sigma not found. Variable", \
            str.upper(var), "not recognized")
            return float('nan')
    
    def get_twiss(self, var, mask = None):
        """Return Twiss parameters (a,b, unnormalized, rms e) for specified coords x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """

        lc_var = var.lower()
        if lc_var in EMITTANCE:
            v = lc_var
            vp = v + 'p'
            if v == 'z':
                v = 'phi'
                vp = 'w'
            avgv2 = math.pow(self.get_sig(v, mask), 2)
            avgvp2 = math.pow(self.get_sig(vp, mask), 2)
            avgvvp = np.mean((self.get_coor(v, mask) - self.get_avg(v, mask)) *
                               (self.get_coor(vp, mask) - self.get_avg(vp, mask)))
            ermssq = avgv2 * avgvp2 - avgvvp * avgvvp
            if ermssq > 0:
                erms = math.sqrt(ermssq)
                alpha = - avgvvp / erms
                beta = avgv2 / erms
                return [alpha, beta, erms]            
            else:
                #print "Error: " + str.upper(var) + " emittance undefined"
                return 3*[float('NaN')]

    def get_urms_emit(self, var, mask = None):
        """Return unnormalized rms emittance along specified axis, x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        a, b, e_urms = self.get_twiss(var, mask)
        return e_urms

    def get_nrms_emit(self, var, mask = None):
        """Return normalized rms emittance along specified axis, x, y or z.

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           mask (Numpy vector, optional): mask used to select particles
        """
        if var in ['x', 'y']:
            return self.get_urms_emit(var, mask) * self.get_betagamma(mask)
        else:
            return self.get_urms_emit(var, mask)

    def get_mask_with_limits(self, var, lolim, uplim = None):
        """Creates a a mask, i.e. a Numpy vector of a list of indices, based upon 
        variable x, xp, y, yp, phi, w or losses above lower limit and below 
        optional upper limit. User units

        Arguments:
           var (str): Either 'x', 'xp', 'y', 'yp', 'phi', 'w', or 'losses'
           lolim (double): lower limit, above which, particles are included in mask
           uplim (double, optional): upper limit, below which, particles are included
           in mask.
        """

        lc_var = var.lower()
        request = "self.get_" + str(lc_var)
        if lc_var in COORDINATES:
            request += "('all')"
        elif lc_var in LOSSES:
            request += "()"
        else:
            print("Error: Empty mask returned, variable", str.upper(var), "not recognized")
            return np.array([])

        aray = eval(request)
        # create test string for later eval in np.where function
        test = "(aray > " + str(lolim) + ")"
        if uplim is not None:
            test += " & (aray < " + str(uplim) + ")"

        return np.where(eval(test))[0]

   
    def get_good_mask(self, mask = None):
        """Returns indices of particles not lost.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        # get_losses = 0 if particle is still good, else element number where it was lost
        if mask is not None:
            return np.intersect1d(mask, np.where(self.beam.get_losses() == 0)[0])
        else:
            return np.where(self.beam.get_losses() == 0)[0]

    def get_lost_mask(self, mask = None):
        """Returns indices of particles lost.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        # get_losses = 0 if particle is still good, else element number where it was lost
        if mask is not None:
            return np.intersect1d(mask, np.where(self.beam.get_losses() != 0)[0])
        else:
            return np.where(self.beam.get_losses() != 0)[0]

    def get_intersection_mask(self, mask1, mask2):
        """Returns the mask that results from the intersection of two masks.
        Arguments:
           mask1 (Numpy vector): mask with condition 1 used to select particles
           mask2 (Numpy vector): mask with condition 2 used to select particles
        """

        return np.intersect1d(mask1, mask2)
        
    def get_betagamma(self, mask = None):
        """Return value of beta*gamma of beam.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        #if mask == None:
        #    gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
        #    return  math.sqrt(gamma * gamma -1.0)
        #elif list(mask) == []:
        #    return 0.0
        #else:
        #    gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
        #    return  math.sqrt(gamma * gamma -1.0)
        if mask is not None:
            if list(mask) == []:
                return 0.0
            else:
                gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
                return  math.sqrt(gamma * gamma -1.0)
        else:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            return  math.sqrt(gamma * gamma -1.0)
            
    def get_betalambda(self, mask = None):
        """Return value of beta*lambda of beam.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        c = 2.99792458e8 # m/s
        wavelength = c / (self.get_frequency() * 1.0e6)
        #if mask == None:
        #    gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
        #    beta = math.sqrt(1.0 - 1/(gamma * gamma))
        #elif list(mask) == []:
        #    beta = 0.0
        #else:
        #    gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
        #    beta = math.sqrt(1.0 - 1/(gamma * gamma))
        #return beta * wavelength
        if mask is not None:
            if list(mask) == []:
                beta = 0.0
            else:
                gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
                beta = math.sqrt(1.0 - 1/(gamma * gamma))
        else:
            gamma = 1.0 + self.get_avg('w', mask)/self.get_mass()
            beta = math.sqrt(1.0 - 1/(gamma * gamma))
            
        return beta * wavelength

    #def get_current(self):
    #    """Return beam current, mA"""
    #    return self.beam.get_current() * MILLIAMP

    def get_current(self, mask = None):
        """Return beam current in user units of beam.
        The original HPSim get_current returns the remaining beam
        current associated with the 'good' particles
        need to scale this result
        """
        #if mask is None:
        #if not isinstance(mask, list):
        #    return self.beam.get_current() * MILLIAMP
        #else:
        #    return self.initial_current * MILLIAMP *  \
        #        self.get_frequency() / self.initial_frequency *  \
        #        self.get_size(mask) / self.initial_gd_size

        if mask is not None:
            return self.initial_current * MILLIAMP *  \
                self.get_frequency() / self.initial_frequency *  \
                self.get_size(mask) / self.initial_gd_size
        else:
            return self.beam.get_current() * MILLIAMP
        
    def get_size(self, mask = None):
        """Return number of beam particles with or w/o mask applied
           Without a mask: Returns number of 'good' particles, i.e. not lost
           With a mask: Returns the length of the mask, i.e. number that satisfy mask.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        #if mask == None:
        #    # return number of good particles, i.e. not lost transversely
        #    # when mask in not present
        #    return self.get_initial_size() - self.beam.get_loss_num()
        #else:
        #    #return length of mask - numpy array when mask is present
        #    return mask.size
        if mask is not None:
            #return length of mask - numpy array when mask is present
            return mask.size
        else:
            # return number of good particles, i.e. not lost transversely
            # when mask in not present
            return self.get_initial_size() - self.beam.get_loss_num()

    def print_results(self, mask = None):
        """Prints avg, sigma, alpha, beta, Eurms, Enrms for all coord of distr.

        Arguments:
           mask (Numpy vector, optional): mask used to select particles
        """
        print("Distribution Analysis Results (w/user units)")
        print('Mass = {0:.4f}'.format(self.get_mass()))
        print('Charge/|e| = {0:.0f}'.format(self.get_charge()))
        print('Ib = {0:.2f} {1}'.format(self.get_current(mask), USER_LABELS['i']))
        print('Frequency = {0:.3f} MHz'.format(self.get_frequency()))
        if mask is not None:
            #if isinstance(mask, list):
            print('*** Mask applied ***')
            print('Number of macroparticles(mask) = {0:.0f}'.format(self.get_size(mask)))
            print('Number of macroparticles lost  = {0:.0f}'.format(self.beam.get_loss_num()))
        else:
            print('*** No Mask applied ***')
            print('Number of macroparticles(good) = {0:.0f}'.format(self.get_size()))
            print('Number of macroparticles lost  = {0:.0f}'.format(self.beam.get_loss_num()))

        print('Ref part. \n phi = {0:10.4f} {1}\n   w = {2:10.4f} {3}'.\
            format(self.get_ref_phi(), USER_LABELS['phi'],
                   self.get_ref_w(), USER_LABELS['w']))
        if self.get_size(mask) > 0:
            print('\nCentroids and RMS sizes')
            print('            Avg         Sigma')
            for item in COORDINATES:
                print('{0:3}: {1:12.4f}    {2:10.4f} {3:3}'\
                    .format(item, self.get_avg(item, mask),
                        self.get_sig(item, mask), USER_LABELS[item]))
            print('\nTwiss parameters')

            print('          Alpha       Beta       Eurms       Enrms')
            for item in EMITTANCE:
                a, b, eurms = self.get_twiss(item, mask)
                if item != EMITTANCE[-1]:
                    print('{0:2}: {1:11.4f} {2:11.4f} {3:11.4f} {4:11.5f}'\
                        .format(item, a, b, eurms, self.get_nrms_emit(item, mask)))
                else: #no normalized long emit
                    print('{0:2}: {1:11.4f} {2:11.4f} {3:11.4f}'\
                        .format(item, a, b, eurms))                 
            print('\n')
        else:
            print('\n*** No particles remaining ****')
            print('\n')
        return
    
############################# Distribution #####################################

class Distribution():
    """An hpsim class for holding a masked beam of particles as a static np-array. 
    Faster for analysis and plotting than using beam array
    """

    coor_index = dict(list(zip(COORDINATES + LOSSES, list(range(0, len(COORDINATES + LOSSES))))))
    def __init__(self, beam, mask = None):
        """Init creates an instance of the Distribution object containing all 
        the vectors of coordinates from the beam object that satisfy the mask

        Attributes:
           mass (double): mc^2 of particle double mass (MeV)
           charge (double): q/|e|
           current (double): bunch peak current(Amps)
           frequency (double): MHz
           size (int): number of macro particles
           betagamma (double): beta * gamma of masked beam
           betalambda (dounble): beta * lambda of masked beam
           ref_phi (double): reference particle phase (Rad)
           ref_w (double): reference particle energy (MeV)

        Returns:
            Returns a beam distribution object

        """
        # create np float array of appropriate size
        if mask is None:
            bm_size = beam.get_initial_size()
        else:
            bm_size = beam.get_size(mask)

        self.coor = np.zeros([len(COORDINATES + LOSSES), bm_size])
        for coor in COORDINATES + LOSSES:
            ndx = Distribution.coor_index[coor]
            self.coor[ndx] = beam.get_coor(coor, mask=mask)

        self.current = beam.get_current(mask)
        self.frequency = beam.get_frequency()
        self.mass = beam.get_mass()
        self.charge = beam.get_charge()
        self.size = beam.get_size(mask)
        self.betagamma = beam.get_betagamma(mask)
        self.betalambda = beam.get_betalambda(mask)
        self.ref_phi = beam.get_ref_phi()
        self.ref_w = beam.get_ref_w()

    def get_ref_phi(self):
        """Return the phase of reference particle"""
        return self.ref_phi

    def get_ref_w(self):
        """Return the phase of reference particle"""
        return self.ref_w

    def get_current(self):
        """Return beam frequency in MHz"""
        return self.current

    def get_frequency(self):
        """Return beam frequency in MHz"""
        return self.frequency

    def get_mass(self):
        """Return mass of beam, mc^2, in MeV"""
        return self.mass

    def get_charge(self):
        """Return charge of beam in q/|e|"""
        return self.charge

    def get_betagamma(self):
        """Return betagamma of beam"""
        return self.betagamma

    def get_betalambda(self):
        """Return betalambda"""
        return self.betalambda

    def get_size(self):
        """Returns total number of beam macro particles"""
        return self.size

    def get_coor(self, var):
        if var in (COORDINATES + LOSSES):
            ndx = Distribution.coor_index[var]
            return self.coor[ndx]
        else:
            print("Error: Empty masked returned, variable", \
            str.upper(var), "not recognized")
            return np.array([])

    def get_loss_num(self):
        """Returns number of macro-particles lost transversely"""
        tloss = self.get_coor('losses')
        return len(tloss[tloss > 0])

    def get_avg(self, var):
        return np.mean(self.get_coor(var))

    def get_sig(self, var):
        return np.std(self.get_coor(var))

    def get_twiss(self, var):
        """Return Twiss parameters (a,b,unnormalized, rms e) for specified coords x, y or z"""
        lc_var = var.lower()
        if lc_var in EMITTANCE:
            v = lc_var
            vp = v + 'p'
            if v == 'z':
                v = 'phi'
                vp = 'w'
            avgv2 = math.pow(self.get_sig(v), 2)
            avgvp2 = math.pow(self.get_sig(vp), 2)
            avgvvp = np.mean((self.get_coor(v) - self.get_avg(v)) *
                               (self.get_coor(vp) - self.get_avg(vp)))
            ermssq = avgv2 * avgvp2 - avgvvp * avgvvp
            if ermssq > 0:
                erms = math.sqrt(ermssq)
                alpha = - avgvvp / erms
                beta = avgv2 / erms
                return [alpha, beta, erms]            
            else:
                #print "Error: " + str.upper(var) + " emittance undefined"
                return 3*[float('NaN')]
        else:
            print("Error: Requested coordinate, " + str.upper(var) + \
                ", must be one of the following:", EMITTANCE)
            return 3*[float('NaN')]       

    def get_urms_emit(self, var):
        """Return unnormalized rms emittance along specified axis, x, y or z"""
        a, b, e_urms = self.get_twiss(var)
        return e_urms

    def get_nrms_emit(self, var):
        """Return normalized rms emittance along specified axis, x, y or z"""
        if var in ['x', 'y']:
            return self.get_urms_emit(var) * self.get_betagamma()
        else:
            return self.get_urms_emit(var)

    def print_results(self):
        """Prints avg, sigma, alpha, beta, Eurms, Enrms for all coord of distr"""
        print("Distribution Analysis Results (w/user units)")
        print('Mass = {0:.4f}'.format(self.get_mass()))
        print('Charge/|e| = {0:.0f}'.format(self.get_charge()))
        print('Ib = {0:.2f} {1}'.format(self.get_current(), USER_LABELS['i']))
        print('Number of macroparticles = {0:.0f}'.format(self.get_size()))
        print('Frequency = {0:.3f} MHz'.format(self.get_frequency()))
        print('*** Mask may have been applied to create Distribution object ***')
        print('Number of macroparticles(in distrubution object) = {0:.0f}'.format(self.get_size()))
        print('Number of macroparticles lost (in distribution object) = {0:.0f}'.format(self.get_loss_num()))
        print('Ref part. \n phi = {0:10.4f} {1}\n   w = {2:10.4f} {3}'.\
            format(self.get_ref_phi(), USER_LABELS['phi'],
                   self.get_ref_w(), USER_LABELS['w']))
        if self.get_size() > 0:
            print('\nCentroids and RMS sizes')
            print('            Avg         Sigma')
            for item in COORDINATES:
                print('{0:3}: {1:10.4f}    {2:10.4f} {3:3}'\
                    .format(item, self.get_avg(item),
                            self.get_sig(item), USER_LABELS[item]))
            print('\nTwiss parameters')
            print('         Alpha      Beta      Eurms      Enrms')
            for item in EMITTANCE:
                a, b, eurms = self.get_twiss(item)
                print('{0:2}: {1:10.4f} {2:10.4f} {3:10.4f} {4:11.5f}'\
                    .format(item, a, b, eurms, self.get_nrms_emit(item)))
            print('\n')
        else:
            print('\n*** No particles remaining ****')
            print('\n')
        return 
        
############################ DBConnection  #####################################
