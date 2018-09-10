# run with python2.7

import numpy as np
from DateTime import DateTime

Date 		= np.genfromtxt('ksiboo_date_sqrt_p.dat', dtype = None)
Date_new 	= [ Date[i][0] + ' ' + Date[i][1] for i in range(len(Date))]
BJD 		= [DateTime(Date_new[i]).timeTime()/(24.*3600) for i in range(len(Date))]

np.savetxt('ksiboo_BJD_sqrt_p.dat', BJD)