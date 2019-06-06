# package
from glob import glob
import pandas as pd
import numpy as np
import os, cv2, sys

from reader import readShortVideo
from model.FVrnnVGG import FVrnnVGG


# os variables
video_dir_path = sys.argv[1]     # directory of full length validation/test videos folder (path to FullLengthVideos/videos/valid )
output_dir = sys.argv[2]         # directory of output labels folder (ex. ./output )


# import valid data
task = "valid"

names = []
train_x = []

for video_path in glob(os.path.join(video_dir_path, '*')) :
    name = video_path.split('/')[-1].replace('.txt','')
    print('[INFO] load images from video :', name, '...', end='')
    
    # load images paths
    img_paths = glob(os.path.join(video_path, '*'))
    img_paths.sort()
        
    # load images
    images = []
    for img_path in img_paths:
        images.append(cv2.imread(img_path))
    images = np.array(images)
    images = images[:,:,:,::-1]
    
    # finally put into train_x and train_y
    names.append(name)
    train_x.append(images)
    print(images.shape, end='  ')
    print('finish') 
    
    
train_x = np.array(train_x)


# load model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FVrnnVGG()

model.to(device)
model.eval()
model.load_train_pretrain()

# eval
from torch import FloatTensor
from torch.autograd import Variable


for idx, (X, name) in enumerate(zip(train_x, names)):
    with open(os.path.join(output_dir, name+'.txt'), 'w+') as file:
        X = np.transpose(X, (0,3,1,2))

        interval = 20
        h = model.h0
        for idx in range(0, X.shape[0])[::interval]:
            x = X[idx:idx+interval]

            x = Variable(FloatTensor(x.copy())).to(device)
            pred, h = model(x, h)
            
            pred = pred.cpu().detach().numpy() # (interval, 11)
            for p in pred:
                file.write(str(p.argmax())+'\n')
    print('[INFO] write prediction into %s successfully' % (name+'.txt'))


# output


