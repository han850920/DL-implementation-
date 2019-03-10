import numpy as np
import math


class cl(object):
    def __init__(self,
                 bs,
                 i_ch,#input channel
                 i_h,#input height
                 i_w,#input width
                 k_num,#kernel number equals to number of output channel
                 k_h,#kernel height
                 k_w,#kernel width
                 stride = 1, # conv move #stride per step
                 zp = 0,#zero padding
                 maxPoolingHeight = 2,
                 maxPoolingWidth = 2,
                 leakyrate = 0.1,
                 activation_func = "LeakyReLu"
                ):
        self.bs = bs
        self.i_ch = i_ch
        self.i_h = i_h
        self.i_w = i_w
        self.k_num = k_num
        self.k_h = k_h
        self.k_w = k_w
        self.stride = stride
        self.zp = zp
        self.maxPoolingHeight = maxPoolingHeight
        self.maxPoolingWidth = maxPoolingWidth
        self.leakyrate = leakyrate
        self.activation_func = activation_func
        
        self.ai = np.zeros((bs, i_ch, i_h+2*zp, i_w+2*zp), dtype = 'float64')
        self.z = np.zeros((bs, k_num, (i_h - k_h + 1)//stride, (i_w - k_w + 1)//stride), dtype = 'float64') # why should i mod stride?
        self.bnz =  np.zeros(self.z.shape,dtype = 'float64') # is it essential?
        self.r =  np.zeros(self.z.shape,dtype = 'float64') # the result after activation
        self.max_pos01 = np.zeros(self.r.shape,dtype = 'int32') # record the maxpooling position, set it by 1 ,others by 0
        self.ao = np.zeros((bs, k_num, (i_h - k_h + 1)//stride//self.maxPoolingHeight, (i_w - k_w + 1)//stride//self.maxPoolingWidth), dtype = "float64")
        self.W = np.random.randn(k_num, i_ch, k_h, k_w).astype("float64")
        self.b = np.random.randn(k_num,1).astype("float64")
        
        self.gamma = np.ones(self.z.shape[1:], dtype='float64')
        self.beta = np.zeros(self.z.shape[1:], dtype='float64')
        
        self.dJdai = np.zeros(self.ai.shape, dtype='float64')
        self.dJdz = np.zeros(self.z.shape, dtype='float64')
        self.dJdbnz = np.zeros(self.bnz.shape, dtype='float64')
        self.dJdr = np.zeros(self.r.shape, dtype='float64')
        self.dJdao = np.zeros(self.ao.shape, dtype='float64')
        self.dJdW = np.zeros(self.W.shape, dtype='float64')
        self.dJdb = np.zeros(self.b.shape, dtype='float64')
        self.dJdgamma = np.zeros(self.gamma.shape, dtype='float64')
        self.dJdbeta = np.zeros(self.beta.shape, dtype='float64')
                           
    def conv_2d(self):
        (i_bs, i_ch, i_rows, i_cols) = self.ai.shape # Input data
        (k_num, k_ch, k_rows, k_cols) = self.W.shape # Kernels
        (z_bs, z_ch, z_rows, z_cols) = self.z.shape   # Output z_ch == k_num
        s = self.stride

        for b in range(z_bs):
            for c in range(z_ch):
                for i in range(z_rows):
                       for j in range(z_cols):
                            self.z[b,c,i,j] = np.sum(np.multiply(self.W[c,:,:,:],
                                                    self.ai[b, : , i*s:i*s+k_rows , j*s:j*s+k_cols ])
                                                   )+self.b[c]
        return
    def conv_prime(self):
        (z_bs, z_ch, z_rows, z_cols) = self.dJdz.shape
        (a_bs, a_ch, a_rows, a_cols) = self.dJdai.shape
        (k_num, k_ch, k_rows, k_cols) = self.dJdW.shape
        s = self.stride # self.stride
        self.dJdai.fill(0.0)
        self.dJdW.fill(0.0)
        self.dJdb.fill(0.0)

        for b in range(z_bs):
            for c in range(z_ch):
                for i in range(z_rows):
                    for j in range(z_cols):
                        dJdz_value = self.dJdz[b, c, i, j]
                        self.dJdai[b, :, i*s:i*s+k_rows, j*s:j*s+k_cols] += dJdz_value * self.W[c, :, :, :]
        for k in range(k_num):
            for c in range(k_ch):
                for i in range(k_rows):
                    for j in range(k_cols):
                        self.dJdW[k,c,i,j] = np.sum(self.ai[:,c,i:i+z_rows*s:s,j:j+z_cols*s:s] *
                                                    self.dJdz[:,k,:,:])/self.bs
        for c in range(z_ch):
            self.dJdb[c] = np.sum(self.dJdz[:,c,:,:])/self.bs

        return

    # All activation functions produce  z --> zr  
    def activation(self):
        if self.activation_func == "tanh":
            self.tanh()
        elif self.activation_func == "swish":
            self.swish()
        elif self.activation_func == "sigmoid":
            self.sigmoid()
        else:
            self.LeakyReLU()
    def activation_prime(self):
        if self.activation_func == "tanh":
            self.tanh_prime()
        elif self.activation_func == "swish":
            self.swish_prime()
        elif self.activation_func == "sigmoid":
            self.sigmoid_prime()
        else:
            self.LeakyReLU_prime()

    def LeakyReLU(self): # if z>0 then r = 1.0*self.bnz else r = self.lekyrate * self.bnz -> 即使小於等於0也值輸出 避免神經元死亡 
        np.copyto(self.r, np.where(self.z > 0, 1.0 * self.bnz, self.leakyrate * self.bnz))
        return

    def LeakyReLU_prime(self): 
        np.copyto(self.dJdbnz, np.where(self.z > 0, 1.0 * self.dJdr, self.leakyrate * self.dJdr))
        return

    def tanh(self):#會有梯度消失的問題
        np.copyto(self.r, (np.exp(self.zr) - np.exp(-self.zr)) / (np.exp(self.zr) + np.exp(-self.zr)))
        return

    def tanh_prime(self):
        np.copyto(self.dJdzr, (1.0 - self.r ** 2))
        return

    def swish(self): #結果比ReLU好 但效能比ReLU略差
        np.copyto(self.zr, self.z * self.sigmoid(self.z))
        return

    def swish_prime(self):
        np.copyto(self.dJdz,
                  self.zr + self.sigmoid(self.z) * (1.0 - self.zr))
        return

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def batch_norm(self, eps=0.01):  
        
        x = self.z
        bs = x.shape[0]
        mu = 1.0/bs * np.sum(x, axis=0)[None, :, :, :]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=0)[None, :, :, :]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        
        np.copyto(self.bnz,
                  self.gamma * xhat + self.beta)
        
        return
    
    # (dJdr, zr, gamma) --> (dJdbeta, dJdgamma, dJdzr)
    def batch_norm_prime(self, eps = 0.01):
        
        x = self.z
        bs = x.shape[0]
        mu = 1.0/bs * np.sum(x, axis=0)[None, :, :, :]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=0)[None, :, :, :]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        
        self.dJdbeta = np.mean(self.dJdbnz, axis=0)[None, :, :, :]
        self.dJdgamma = np.mean(self.dJdbnz * xhat, axis=0)[None, :, :, :]
        self.dJdz = (1.0 - 1.0/bs) * (1.0/v - u**2/v**3/bs) * self.gamma * self.dJdbnz
        
        return 
    
    def batch_norm_pass(self, eps=0.01):
        
        np.copyto(self.r, self.zr)
        
        return
    
    def batch_norm_prime_pass(self, eps = 0.01):
        
        np.copyto(self.dJdzr, self.dJdr)     
        return   


    def maxPool(self):
        (r_bs, r_ch, r_rows, r_cols) = self.r.shape
        (a_bs, a_ch, a_rows, a_cols) = self.ao.shape
        self.max_pos01.fill(0) # record from which is the maximum so we can backprop
        for b in range(a_bs):
            for c in range(a_ch):
                for i in range(a_rows):
                    for j in range(a_cols):
                        pooling_src = self.r[b,
                                             c,
                                             i*self.maxPoolingHeight:(i+1)*self.maxPoolingHeight,
                                             j*self.maxPoolingWidth:(j+1)*self.maxPoolingWidth]
                        self.ao[b,c,i,j] = np.max(pooling_src)
                        max_pos = np.unravel_index(np.argmax(pooling_src),np.shape(pooling_src))
                        self.max_pos01[b, c, i*self.maxPoolingHeight+max_pos[0], j*self.maxPoolingWidth+max_pos[1]] = 1
        return
    #computing dJdr
    def maxPool_prime(self):
        (r_bs,r_ch,r_rows, r_cols)=self.dJdr.shape
        for b in range(r_bs):
            for c in range(r_ch):
                for i in range(r_rows):
                    for j in range(r_cols):
                        self.dJdr[b,c,i,j] = (self.dJdao[b,c,i//self.maxPoolingHeight,j//self.maxPoolingWidth]
                                              if self.max_pos01[b,c,i,j] ==1
                                              else 0.0)
        return


    # a(i-1) --conv--> z --activ--> r --maxpool--> a(i)
    def forward(self):
        self.conv_2d()
        self.batch_norm()
        self.activation()
        self.maxPool()
        return
    def backprop(self):
        self.maxPool_prime()
        self.activation_prime()
        self.batch_norm_prime()
        self.conv_prime()
        return
    def update(self,lr,eta):
        self.W = (1.0 - eta * lr) * self.W - lr * self.dJdW
        self.b = (1.0 - eta * lr) * self.b -lr * self.dJdb
        self.gamma = (1.0 - eta * lr) * self.gamma -lr * self.dJdgamma
        self.beta = (1.0 - eta * lr) * self.beta -lr * self.dJdbeta

