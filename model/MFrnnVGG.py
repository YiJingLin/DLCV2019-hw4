import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

import copy

# Multi Frame VGG model
class MFrnnVGG(nn.Module):
    def __init__(self, backend='vgg16', pretrained=True, n_label=11):
        super(MFrnnVGG, self).__init__()
        
        ### check valid 
        if backend in ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn']:
            pass
        else :
            print("[INFO] invalid backend '%s', change to 'vgg16_bn'" % backend)
            backend = 'vgg16_bn'
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ### init param
        self.backend = backend
        self.pretrained = pretrained
        # model flow
        self.features = None
        self.avgpool = None
        self.RNN = None
        self.h0 = None # follow RNN
        self.c0 = None # follow RNN
        self.classifier = None
        
        ### init process
        self.load_pretrained() # load features
        self.create_RNN() # create RNN 
        self.create_classifier(n_label) # create last layer
        self.fix_features() # fix features weights
        
    def forward(self, input):
        '''
        input shape : (frame, channel, height, weight)
        output shape : (1, cls)
        '''
        f, c, h, w = input.shape
        
        # regard f:frames as b:batch
        x = self.features(input) # shape : (f, 512, 7, 10)
        x = self.avgpool(x) # shape (f, 512, 7, 7)      
        
        x = torch.flatten(x, start_dim=1) # (f, 25088)
        x = torch.unsqueeze(x,0) # (1, f, 25088)
        
        out, h = self.RNN(x, self.h0) # out(1, f, 1024) & (num_layers=1, 1, 1024)
        x = torch.squeeze(h, 0) # (1, 1024)        
        
        x = self.classifier(x) # out shape : (1, 11)
        return x
    
    def load_pretrained(self):
        import torchvision.models as models
        backend_model = None
        try:
            if self.backend == 'vgg13' :
                backend_model = models.vgg13(pretrained=self.pretrained)
            elif self.backend == 'vgg13_bn' :
                backend_model = models.vgg13_bn(pretrained=self.pretrained)
            elif self.backend == 'vgg16' :
                backend_model = models.vgg16(pretrained=self.pretrained)
            elif self.backend == 'vgg16_bn':
                backend_model = models.vgg16_bn(pretrained=self.pretrained)
            
            
            else :
                raise ValueError("[ERROR] Unexpected backend name pass through previous check then into load_pretrained() .")
            # copy features flow
            self.features = copy.deepcopy(backend_model.features) 
            self.avgpool = copy.deepcopy(backend_model.avgpool)
            print("[INFO] load pretrained features successfully, backend : %s" % self.backend)
        except Exception as e:
            print(e)
    
    def create_RNN(self, rnn='GRU', hidden_size=1024, num_layers=1, batch_first=True):
        '''
        output (batch, seq, hidden_size)
        h_out (n_layer, batch, hidden_size)
        '''
        try:
            input_size = None
            if self.backend in ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn']:
                input_size = 25088
            else :
                raise ValueError("[ERROR] Unexpected backend name pass through previous check then into create_outLayer() .")
            
            if rnn == 'GRU' :
                self.RNN = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers, 
                    batch_first=batch_first,
                )
                self.h0 = Variable(torch.zeros((num_layers,1,hidden_size)), requires_grad=False).to(self.device) # bach_size = 1
            
            else :
                raise ValueError("[ERROR] Unexpected rnn '%s', please select one in ['GRU']" & rnn)
                
            print("[INFO] create RNN component successfully, rnn : %s ." % rnn)
        except Exception as e:
            print(e)
        
        
    def create_classifier(self, n_label=11):
        try:
            if self.backend in ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn'] :
                self.classifier = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 11),
                    nn.Softmax(),
                )
            else :
                raise ValueError("[ERROR] Unexpected backend name pass through previous check then into create_outLayer() .")
        
            print("[INFO] create classifier successfully.")
        except Exception as e:
            print(e)
                
    def fix_features(self): # fix features weights
        for param in self.features.parameters():
            param.requires_grad = False
    
    def load_train_pretrain(self):
        if self.device == 'cuda':
            self.RNN.load_state_dict(torch.load('./storage/MFrnnVGG_RNN.pkl'))
            self.classifier.load_state_dict(torch.load('./storage/MFrnnVGG_classifier.pkl'))
        else :
            self.RNN.load_state_dict(torch.load('./storage/MFrnnVGG_RNN.pkl', map_location=lambda storage, loc: storage))
            self.classifier.load_state_dict(torch.load('./storage/MFrnnVGG_classifier.pkl', map_location=lambda storage, loc: storage))
        
        print("[INFO] load train pretrained weight successfully.")
            

def main():
    model = MFrnnVGG(backend='vgg13_bn', pretrained=False)
    model.load_train_pretrain()
    model
    
    
if __name__ == "__main__" :
    main()
