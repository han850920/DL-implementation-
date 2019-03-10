import numpy as np
from model.conv_layer import cl 


class cnn(object):
    def __init__(self,
                input_data_spec = [10,3,28,28],
                conv_layer_spec = [{"k_num":2,
                                    "k_h":3,
                                    "k_w":3,
                                    "stride":1,
                                    "zp":0,
                                    "maxPoolingHeight":2,
                                    "maxPoolingWidth":2}],
                fc_layer_spec=[100,50,10]):
        
        #conv layer list
        self.cnn_net = []
        for i in range(len(conv_layer_spec)):
            self.cnn_net.append(cl(bs = input_data_spec[0],
                                           i_ch = input_data_spec[1],
                                           i_h = input_data_spec[2],
                                           i_w = input_data_spec[3],
                                           k_num = conv_layer_spec[i]["k_num"],
                                           k_h = conv_layer_spec[i]["k_h"],
                                           k_w = conv_layer_spec[i]["k_w"],
                                           stride = conv_layer_spec[i]["stride"],
                                           zp = conv_layer_spec[i]["zp"],
                                           maxPoolingHeight = conv_layer_spec[i]["maxPoolingHeight"],
                                           maxPoolingWidth = conv_layer_spec[i]["maxPoolingWidth"]))
            input_data_spec = list(self.cnn_net[i].ao.shape)
        
        
        (bs,ch,r,c) = self.cnn_net[-1].ao.shape # last layer of CNN and connect to fully-connected layer
        fc_layer_spec[0] = ch * r * c
        self.fc_net = MLP(Layers = fc_layer_spec, BatchSize = input_data_spec[0])
        return
    
    def forward(self,input_data):
        
        #forward the net
        for i in range(len(self.cnn_net)):
            if i == 0:
                np.copyto(self.cnn_net[i].ai,input_data)
            else:
                np.copyto(self.cnn_net[i].ai, self.cnn_net[i-1].ao)
            self.cnn_net[i].forward()
        
        #flatten the last layer and forwad to MLP 
        input_data_fc = np.copy(self.cnn_net[-1].ao.reshape(self.cnn_net[-1].ao.shape[0],-1).T)
        self.fc_net.forward(input_data_fc)
    
    def backprop(self):
        
        #MLP backprop
        self.fc_net.backprop()
        #transfer the gradient from MLP to the CNN
        np.copyto(self.cnn_net[-1].dJdao,
                  self.fc_net.net[0].dJdi.T.reshape(self.cnn_net[-1].dJdao.shape))
        #CNN backprop
        for i in range(len(self.cnn_net)-1,-1,-1):
            self.cnn_net[i].backprop()
            
            if i>0:
                np.copyto(self.cnn_net[i-1].dJdao,self.cnn_net[i].dJdai)
        return
    
    def update(self,lr,eta):
        #update the weight of every conv layer
        for i in range(len(self.cnn_net)):
            self.cnn_net[i].update(lr,eta)
        self.fc_net.update(lr,eta)
        return
    
    def train(self, train_x, train_y, val_x, val_y, epoch_cnt, lr, eta):
        for e in range(epoch_cnt):
            #shufle the input_data
            shuffle = np.arange(train_x.shape[0])#得一個 0~len(train_x)-1 且 順序打亂的序列
            train_x_s = train_x[shuffle] #再照順序給 可得shuffle後的結果
            train_y_s = train_y[shuffle]
            bs = self.fc_net.bs
            for i in range(train_x_s.shape[0]//bs):
                
                x = train_x_s[i*bs:(i+1)*bs,:,:,:]
                y = train_y_s[i*bs:(i+1)*bs]
                self.forward(x)
                self.fc_net.loss(y,eta)
                self.backprop()
                self.update(lr,eta)

                
                #output the procedure
                print("\nEpoch:",e,"Batch: ",i, "J=",self.fc_net.J[-1],
                      "Error Rate: ",np.count_nonzero(np.array(y-self.fc_net.yhat)/float(len(y))))
                print("\n Proper learn rate: ",lr )
                print ("\nMax abs(a) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[-1].a)), 
                       "\nMax abs(dJda) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[-1].dJda)),
                      "\nMax abs(W) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[1].W)), "\n")
            
            #Validation
            for i in range(val_x.shape[0]//bs):
                x = val_x[i*bs:(i+1)*bs,:,:,:]
                y = val_y[i*bs:(i+1)*bs]
                self.forward(x)
                self.fc_net.loss_val(y,eta)
        return  

