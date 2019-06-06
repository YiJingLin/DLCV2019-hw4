# package
from glob import glob
import pandas as pd
import numpy as np
import os, cv2, sys

from reader import readShortVideo
from model.MFVGG import MFVGG


# os variables
video_dir_path = sys.argv[1]     # directory of trimmed validation/test videos folder (ex. path to TrimmedVideos/video/valid )
csv_path = sys.argv[2]           # path of the gt_valid.csv/gt_test.csv (ex. path to TrimmedVideos/label/gt_valid.csv )
output_dir = sys.argv[3]         # directory of output labels folder (ex. ./output )


# import valide table
train_table = pd.read_csv(csv_path)


# valid data
train_x = []
train_y = []

# load valid data
for idx, value in train_table[['Video_category', 'Video_name', 'Action_labels']].iterrows() :
    
    video_category = value.Video_category
    video_name = value.Video_name

    def custom_VideoNameExtractor(video_dir_path, video_category, video_name):
        video_name = glob(os.path.join(video_dir_path, video_category, video_name)+'*')[0]
        video_name = video_name.split('/')[-1]
        return video_name
    video_name = custom_VideoNameExtractor(video_dir_path, video_category, video_name)
    
    try:
        
        frames =  readShortVideo(video_path=video_dir_path, 
                                 video_category=video_category, 
                                 video_name = video_name)
        train_x.append(frames / 255)
        train_y.append(value.Action_labels)
    except Exception as e:
        print(e)
        
    if (idx+1) % 100 == 0 :
        print("[INFO] loading progress, (%s/%s)" % ((idx+1), len(train_table)))
    
print("[INFO] load train_x successfully, train_x length :", len(train_x))


# load model
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFVGG()

model.to(device)
model.eval()
if device == 'cuda':
    model.classifier.load_state_dict(torch.load(os.path.join('./storage/MFVGG_classifier.pkl')))
else :
    model.classifier.load_state_dict(torch.load(os.path.join('./storage/MFVGG_classifier.pkl'), map_location=lambda storage, loc: storage))  
# eval
from torch import FloatTensor
from torch.autograd import Variable

with open(os.path.join(output_dir, 'p1_valid.txt'), 'w+') as file:
    for idx, x in enumerate(train_x):
        try:
            x = np.transpose(x, (0,3,1,2))
            x = Variable(FloatTensor(x)).to(device)
            pred = model(x)
            pred = pred.argmax().item()
            file.write(str(pred)+'\n')
        except Exception as e:
            print(idx, e)
            file.write('1\n')


# output


