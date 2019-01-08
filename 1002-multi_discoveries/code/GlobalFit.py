import numpy as np

#==============================================================================
# GlobalFit without a planet
#==============================================================================
class GlobalFit(object):
    '''
    calculate the rotational activity fit of different years
    '''

    A_RV_Rhk    = 66.1781

    def activity2009(self, t, A_RV_Rhk=A_RV_Rhk, RHK_l=RHK_l):
        '''
        2009 activity
        '''        
        P1          = 39.7572
        A11s        = 0.5022
        A11c        = 1.3878
        A12s        = 0.7684
        A12c        = 0.2643

        self.iddx1  = (t < 54962) & (t > 54872)
        RHK_l_2009  = RHK_l[self.iddx1]
        x           = t[self.iddx1] - 55279.109840075726
        print(x)

        # return A_RV_Rhk*RHK_l_2009 + \
        # A11s*np.sin(2*np.pi/P1*x) + A11c*np.cos(2*np.pi/P1*x) + \
        # A12s*np.sin(2*np.pi*2/P1*x) + A12c*np.cos(2*np.pi*2/P1*x) 

    def activity2010(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        2010 activity
        '''        
        P2          = 37.8394
        A21s        = -1.0757
        A21c        = 1.1328 
        A23s        = -1.3124
        A23c        = -1.0487
        A24s        = -0.1096
        A24c        = -1.3694

        iddx2   = (t < 55363) & (t > 55273)
        RHK_l_2010  = RHK_l[iddx2]
        x           = t[iddx2] - 55279.109840075726
        
        return A_RV_Rhk*RHK_l_2010 + \
        A21s*np.sin(2*np.pi/P2*x) + A21c*np.cos(2*np.pi/P2*x) + \
        A23s*np.sin(2*np.pi*3/P2*x) + A23c*np.cos(2*np.pi*3/P2*x) + \
        A24s*np.sin(2*np.pi*4/P2*x) + A24c*np.cos(2*np.pi*4/P2*x)

    def activity2011(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        2011 activity
        '''
        P3          = 36.7549
        A31s        = 1.1029
        A31c        = -0.9084
        A32s        = -0.7422
        A32c        = -0.3392
        A33s        = -1.2984
        A33c        = 0.707

        iddx3       = (t < 55698) & (t > 55608)
        RHK_l_2011  = RHK_l[iddx3]
        x           = t[iddx3] - 55279.109840075726

        return A_RV_Rhk*RHK_l_2011 + \
        A31s*np.sin(2*np.pi/P3*x) + A31c*np.cos(2*np.pi/P3*x) + \
        A32s*np.sin(2*np.pi*2/P3*x) + A32c*np.cos(2*np.pi*2/P3*x) + \
        A33s*np.sin(2*np.pi*3/P3*x) + A33c*np.cos(2*np.pi*3/P3*x)

    def binary(self, t):
        '''
        binary orbit 
        '''
        lin0        = -22700.1747
        lin1        = -0.5307
        lin2        = -1.83e-5
        BJD0        = 55279.109840075726
        x           = np.hstack((t[(t < 54962) & (t > 54872)], t[(t < 55363) & (t > 55273)], t[(t < 55698) & (t > 55608)]))
        return lin0 + lin1 * (x-BJD0) + lin2 * (x-BJD0)**2

    def activity_all(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        all three years
        '''
        return np.hstack((self.activity2009(t, A_RV_Rhk=A_RV_Rhk), 
            self.activity2010(t, A_RV_Rhk=A_RV_Rhk), 
            self.activity2011(t, A_RV_Rhk=A_RV_Rhk)))

    def fit(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        global fit
        '''
        return self.binary(t) + self.activity_all(t, A_RV_Rhk=A_RV_Rhk)

#==============================================================================
# GlobalFit with a planet
#==============================================================================
class GlobalFit2(object):
    '''
    calculate the rotational activity fit of different years
    '''

    A_RV_Rhk    = 65.4884

    def activity2009(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        2009 activity
        '''        
        P1          = 39.7579
        A11s        = 0.6129
        A11c        = 1.4052
        A12s        = 0.8131
        A12c        = 0.2452

        iddx1       = (t < 54962) & (t > 54872)
        RHK_l_2009  = RHK_l[iddx1]
        x           = t[iddx1]

        return A_RV_Rhk*RHK_l_2009 + \
        A11s*np.sin(2*np.pi/P1*x) + A11c*np.cos(2*np.pi/P1*x) + \
        A12s*np.sin(2*np.pi*2/P1*x) + A12c*np.cos(2*np.pi*2/P1*x) 

    def activity2010(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        2010 activity
        '''        
        P2          = 37.7984
        A21s        = -1.0511
        A21c        = 1.199 
        A23s        = -1.3561
        A23c        = -1.0452
        A24s        = -0.1568
        A24c        = -1.4155

        iddx2   = (t < 55363) & (t > 55273)
        RHK_l_2010  = RHK_l[iddx2]
        x           = t[iddx2]
        
        return A_RV_Rhk*RHK_l_2010 + \
        A21s*np.sin(2*np.pi/P2*x) + A21c*np.cos(2*np.pi/P2*x) + \
        A23s*np.sin(2*np.pi*3/P2*x) + A23c*np.cos(2*np.pi*3/P2*x) + \
        A24s*np.sin(2*np.pi*4/P2*x) + A24c*np.cos(2*np.pi*4/P2*x)

    def activity2011(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        2011 activity
        '''
        P3          = 36.7061 
        A31s        = 1.0425
        A31c        = -1.0449
        A32s        = -0.7776
        A32c        = -0.2521 
        A33s        = -1.0378 
        A33c        = 1.0088

        iddx3       = (t < 55698) & (t > 55608)
        RHK_l_2011  = RHK_l[iddx3]
        x           = t[iddx3]

        return A_RV_Rhk*RHK_l_2011 + \
        A31s*np.sin(2*np.pi/P3*x) + A31c*np.cos(2*np.pi/P3*x) + \
        A32s*np.sin(2*np.pi*2/P3*x) + A32c*np.cos(2*np.pi*2/P3*x) + \
        A33s*np.sin(2*np.pi*3/P3*x) + A33c*np.cos(2*np.pi*3/P3*x)

    def binary(self, t):
        '''
        binary orbit 
        '''
        lin0        = -22700.1678
        lin1        = -0.5305
        lin2        = 0
        x           = np.hstack((BJD[iddx1], BJD[iddx2], BJD[iddx3]))
        return lin0 + lin1 * (x) + lin2 * (x)**2

    def activity_all(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        all three years
        '''
        return np.hstack((self.activity2009(t, A_RV_Rhk=A_RV_Rhk), 
            self.activity2010(t, A_RV_Rhk=A_RV_Rhk), 
            self.activity2011(t, A_RV_Rhk=A_RV_Rhk)))

    def planet(self, t):
        nu  = 0.309
        A   = 0.4838
        B   = -0.1619
        return A*np.cos(2*np.pi*BJD) + B*np.cos(2*np.pi*BJD)

    def fit(self, t, A_RV_Rhk=A_RV_Rhk):
        '''
        global fit
        '''
        return self.binary(t) + self.activity_all(t, A_RV_Rhk=A_RV_Rhk)        
