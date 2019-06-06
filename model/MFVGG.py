import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

import copy

# Multi Frame VGG model
class MFVGG(nn.Module):
    def __init__(self, backend='vgg16', pretrained=True, n_label=11):
        super(MFVGG, self).__init__()
        
        ### check valid 
        if backend in ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn']:
            pass
        else :
            print("[INFO] invalid backend '%s', change to 'vgg16_bn'" % backend)
            backend = 'vgg16_bn'
        
        ### init param
        self.backend = backend
        self.pretrained = pretrained
        # model flow
        self.features = None
        self.avgpool = None
        self.classifier = None
        self.outLayer = None # customize output for task : Linear(1000, 11)
        
        ### init process
        self.load_pretrained() # load features
        self.create_classifier(n_label) # create last layer
#         self.fix_features() # fix features weights
        
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
        x = torch.mean(x, 0, keepdim=True) # (1, 25088)
        
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
                    
    def create_classifier(self, n_label=11):
        try:
            if self.backend in ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn'] :
                self.classifier = nn.Sequential(
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 1000),
                    nn.Linear(1000, 11),
                    nn.Softmax(),
                )
            else :
                raise ValueError("[ERROR] Unexpected backend name pass through previous check then into create_outLayer() .")
        except Exception as e:
            print(e)
                
    def fix_features(self): # fix features weights
        for param in self.features.parameters():
            param.requires_grad = False
            

def main():
    model = MFVGG(backend='vgg13_bn', pretrained=False)
    model
    
    
if __name__ == "__main__" :
    main()
