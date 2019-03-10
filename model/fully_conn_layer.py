import numpy as np

class fcl():
    def __init__(self,
                 layer,
                 pre_layer,
                 bs = 4):
        self.bs = bs
        self.layer = layer
        self.pre_layer = pre_layer
        self.i = np.zeros((pre_layer,self.bs),dtype="float64")
        self.z = np.zeros((self.layer,self.bs),dtype="float64")
        self.bnz = np.zeros((self.layer,self.bs),dtype="float64")
        self.a = np.zeros((self.layer,self.bs),dtype="float64")
        self.W = np.random.randn(self.layer, self.pre_layer).astype('float64')
        self.b = np.random.randn(self.layer, 1).astype('float64')
        self.gamma = np.random.randn(self.layer,1).astype('float64')
        self.beta = np.random.randn(self.layer,1).astype('float64')
        
        self.dJdi = np.zeros(self.i.shape,dtype='float64')
        self.dJdz = np.zeros(self.z.shape,dtype='float64')
        self.dJdbnz = np.zeros(self.bnz.shape,dtype='float64')
        self.dJda = np.zeros(self.a.shape,dtype='float64')
        self.dJdW = np.zeros(self.W.shape,dtype='float64')
        self.dJdb = np.zeros(self.b.shape,dtype='float64')
        self.dJdgamma = np.zeros(self.gamma.shape,dtype='float64')
        self.dJdbeta = np.zeros(self.beta.shape,dtype='float64')
        

