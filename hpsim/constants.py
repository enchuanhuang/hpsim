import math
#global variables

COORDINATES = ['x', 'xp', 'y', 'yp', 'phi', 'w']
EMITTANCE = ['x', 'y', 'z']
PHASESPACE = ['xxp', 'yyp', 'phiw']
LOSSES = ['losses']
# hpsim C code uses MeV, radians, meters
# units conversion factors so hpsim scripts use MeV, deg, cm, milliradians
CM = 100.0
MILLIRAD = 1000.0
DEG = 180.0/math.pi
RAD = 1/DEG
MILLIAMP = 1000.0 
USER_UNITS = {'x':CM, 'xp':MILLIRAD, 'y':CM, 'yp':MILLIRAD, 'phi':DEG, 'w':1.0,
             'xavg':CM, 'xsig':CM, 'xpavg':MILLIRAD, 
             'xpsig':MILLIRAD, 'xemit':CM*MILLIRAD,
             'yavg':CM, 'ysig':CM, 'ypavg':MILLIRAD, 
             'ypsig':MILLIRAD, 'yemit':CM*MILLIRAD,
             'phiavg':DEG, 'phisig':DEG, 'phiref':DEG, 
             'z':CM,
             'wavg':1.0, 'wsig':1.0, 'wref':1.0, 'zemit':1.0, # FIXME zemit
             'loss_ratio':1, 'loss_local':1,'model_index':1}
USER_LABELS = {'x':'cm', 'xp':'mr', 'y':'cm', 'yp':'mr', 'phi':'deg', 'w':'MeV',
                'i':'mA'}