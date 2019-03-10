import numpy as np
from model.fully_conn_layer import fcl

class mlp(object):
    def __init__(self, Layers=(2, 5, 3), BatchSize = 4, activation="ReLU", leakyRate=0.01):
        self.bs = BatchSize
        self.activation = activation
        self.leakyRate = leakyRate
        self.net=[]
        for i in range(len(Layers)):
            if i == 0 :
                self.net.append(fcl(layer=Layers[0], pre_layer=Layers[0], bs = self.bs))
            else:
                self.net.append(fcl(layer=Layers[i], pre_layer=Layers[i-1], bs = self.bs))
        self.p = np.zeros(self.net[-1].a.shape,dtype='float64')
        self.dJdp = np.zeros(self.p.shape,dtype='float64')
        self.yhat = np.zeros(self.bs,dtype=int)#predict answer, presented by probility
        self.y_predict = np.zeros(self.p.shape,dtype=int)#one-hot encodeed
        
        self.J = [] #Loss value trace
        self.J_val = []#Loss value trace for validation
        self.L2_regularization=[]
              
    def forward(self,x):
        np.copyto(self.net[0].i,x)
        np.copyto(self.net[0].z,self.net[0].i)
        np.copyto(self.net[0].bnz,self.net[0].z)
        np.copyto(self.net[0].a,self.net[0].bnz)
                  
        for i in range(1,len(self.net)):
            np.copyto(self.net[i].i, self.net[i-1].a)
            np.copyto(self.net[i].z, np.dot(self.net[i].W,self.net[i].i) + self.net[i].b)
            np.copyto(self.net[i].bnz,self.batch_norm(self.net[i].z,
                                                      self.net[i].gamma,
                                                      self.net[i].beta,
                                                      0.01))
            self.activation_func(i=i)
        np.copyto(self.p,self.softmax(self.net[-1].a))
        np.copyto(self.yhat,np.argmax(self.p,axis = 0))
       
        return
                  
    # Batch Normalization for an MLP Layer
    # x --> bnx              
    def batch_norm(self, x, gamma, beta, eps):
        
        bs = x.shape[-1]
        mu = 1.0/bs * np.sum(x, axis=-1)[:, None]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=-1)[:, None]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        bnx = gamma * xhat + beta
        
        return bnx
    
    # Backprop Batch Normalization for an MLP Layer
    # (dJdbnx, x, gamma, beta) --> (dJdbeta, dJdgamma, dJdx)
    def batch_norm_prime(self, dJdbnx, x, gamma, beta, eps):
        
        bs = x.shape[-1]
        mu = 1.0/bs * np.sum(x, axis=-1)[:, None]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=-1)[:, None]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        
        dJdbeta = np.mean(dJdbnx, axis=-1)[:, None]
        dJdgamma = np.mean(dJdbnx * xhat, axis=-1)[:, None]
        dJdx = (1.0 - 1.0/bs) * (1.0/v - u**2 / v**3 / bs) * gamma * dJdbnx
        
        return dJdbeta, dJdgamma, dJdx
                  
                  
    # Activation function
    def activation_func(self,i):
        if self.activation == "sigmoid":
            np.copyto(self.net[i].a, self.sigmoid(self.net[i].bnz))
        elif self.activation == "tanh":
            np.copyto(self.net[i].a, self.tanh(self.net[i].bnz))
        elif self.activation == "swish":
            np.copyto(self.net[i].a, self.swish(self.net[i].bnz))
        elif self.activation == "LeakyReLU":
            np.copyto(self.net[i].a, self.LeakyReLU(self.net[i].bnz))
        else:
            np.copyto(self.net[i].a, self.ReLU(self.net[i].bnz))
    def activation_func_prime(self,i):
        if self.activation == "sigmoid":
            np.copyto(self.net[i].dJdbnz,(self.net[i].dJda * self.sigmoidPrime(self.net[i].a))) 
        elif self.activation == "tanh":
            np.copyto(self.net[i].dJdbnz,(self.net[i].dJda * self.tanhPrime(self.net[i].a))) 
        elif self.activation == "swish":
            np.copyto(self.net[i].dJdbnz,(self.net[i].dJda * self.swishPrime(self.net[i].a, self.net[i].bnz)))
        elif self.activation == "LeakyReLU":
            np.copyto(self.net[i].dJdbnz,(self.net[i].dJda * self.LeakyReLUPrime(self.net[i].a,self.leakyRate))) 
        else:
            np.copyto(self.net[i].dJdbnz,(self.net[i].dJda * self.ReLUPrime(self.net[i].a)))       
                  
    def softmax(self, a):
        return np.exp(a) / np.sum(np.exp(a), axis=0)
    
    # Sigmoid activation function: z --> a
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-1.0*z))
    
    # a --> dadz
    def sigmoidPrime(self, a):
        return a * (1.0 - a)
    
    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    def tanhPrime(self, a):
        return (1.0 - a ** 2) # Derivative of tanh is (1.0 - tanh ** 2)
    
    def swish(self, z):
        return z * self.sigmoid(z)
    
    def swishPrime(self, a, z):
        return a + self.sigmoid(z) * (1.0 - a) 
    
    def ReLU(self, z):
        a = np.copy(z)
        a[a<0] = 0.0
        return a
    
    def ReLUPrime(self, a):
        dadz = np.copy(a)
        dadz[a>0] = 1.0
        return dadz
    
    def LeakyReLU(self, z, leakyrate):
        a = np.copy(z)
        (rows, cols) = a.shape
        for i in range(rows):
            for j in range(cols):
                a[i, j] = z[i, j] if z[i,j] > 0 else (leakyrate * z[i, j])
        return a
    
    def LeakyReLUPrime(self, a, leakyrate):
        dadz = np.copy(a)
        (rows, cols) = dadz.shape
        for i in range(rows):
            for j in range(cols):
                dadz[i, j] = 1.0 if a[i,j] > 0 else leakyrate
        return dadz
    
    def loss(self, y, eta):
                  
        self.y_predict.fill(0)
        for i in range(self.bs):
            self.y_predict[y[i],i] = 1
        self.J.append(-1.0 * np.sum(self.y_predict * np.log(self.p) / self.bs))
        
        # L2 Regularization
        L2 = 0.0 
        for i in range(1, len(self.net)):
            L2 += np.sum(self.net[i].W ** 2)              
        self.L2_regularization.append(eta / 2 * L2) # Only MLP do 
        
        return
                  
    # Loss function for validation
    def loss_val(self, y, eta):
        
        self.y_predict.fill(0)
        for i in range(self.bs):
            self.y_predict[y[i], i] = 1     
        self.J_val.append(-1.0 * np.sum(self.y_predict * np.log(self.p) / self.bs))
        
        return
    def backprop(self):
        
        self.dJdp = 1.0 / (1.0 - self.y_predict - self.p)
        dpda = np.array([[self.p[i, :] * (1.0-self.p[j, :]) if i == j
                          else -1 * self.p[i, :] * self.p[j, :]
                          for i in range(self.p.shape[0])]
                         for j in range(self.p.shape[0])])
        for i in range(self.bs):
            self.net[-1].dJda[:, i] = np.dot(dpda[:, :, i], self.dJdp[:, i])
            
        for i in range(len(self.net)-1, 0, -1):
            
            self.activation_func_prime(i=i)
           
            dJdbeta, dJdgamma, dJdBNinput = self.batch_norm_prime(self.net[i].dJdbnz,
                                                                  self.net[i].z,
                                                                  self.net[i].gamma,
                                                                  self.net[i].beta,
                                                                  0.01)  
            np.copyto(self.net[i].dJdbeta, dJdbeta)
            np.copyto(self.net[i].dJdgamma, dJdgamma)
            np.copyto(self.net[i].dJdz, dJdBNinput) 
            
            np.copyto(self.net[i].dJdb,
                      np.mean(self.net[i].dJdz, axis = -1)[:, None])         
            np.copyto(self.net[i].dJdW,
                      np.dot(self.net[i].dJdz, self.net[i].i.T) / self.bs)  
            np.copyto(self.net[i].dJdi,
                      np.dot(self.net[i].W.T, self.net[i].dJdz))
            
            # Copy gradient at the input to be the output gradient of the previous layer
            np.copyto(self.net[i-1].dJda, self.net[i].dJdi)
        
        # Layer 0 does nothing but passing gradients backward
        np.copyto(self.net[0].dJdbnz, self.net[0].dJda)
        np.copyto(self.net[0].dJdz, self.net[0].dJdbnz)
        np.copyto(self.net[0].dJdi, self.net[0].dJdz) 
        return
    # Update parameters
    def update(self, lr, eta):
        
        # Update W, b, gamma, beta with Weight Decay from Layer 1 to the last
        # Layer 0 has no parameters
        for i in range(1, len(self.net)):
            np.copyto(self.net[i].W,
                      (1.0 - eta * lr) * self.net[i].W - lr*self.net[i].dJdW)
            np.copyto(self.net[i].b,
                      (1.0 - eta * lr) * self.net[i].b - lr*self.net[i].dJdb)
            np.copyto(self.net[i].gamma,
                      (1.0 - eta * lr) * self.net[i].gamma - lr*self.net[i].dJdgamma)
            np.copyto(self.net[i].beta,
                      (1.0 - eta * lr) * self.net[i].beta - lr*self.net[i].dJdbeta)
        return
    
    # Train MLP alone. For a CNN, training is via the CNN instance
    def train(self, train_x, train_y, epoch_count, lr, eta):
        
        for e in range(epoch_count):
            # print ("Epoch ", e)
            for i in range(train_x.shape[1]//self.bs):
                x = train_x[:, i*self.bs:(i+1)*self.bs]
                y = train_y[i*self.bs:(i+1)*self.bs]
                self.forward(x)
                self.loss(y, eta)
                self.backprop()
                self.update(lr, eta)           
        return  
    

