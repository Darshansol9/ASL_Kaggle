import numpy as np
import config
import pandas as pd
import joblib
import albumentations
from PIL import Image
import torch
import models
from dispacter import MODEL_DISPATCHER


DEVICE = config.DEVICE
    
class ASIDataSetTest:

    def __init__(self,img_height,img_width,mean,std,file_name=None):

        img_read = Image.open(config.TEST_PATH+f'/{file_name}')

        #adding bs dimension
        self.image = np.asarray(img_read)[None,:,:,:]
        self.aug = albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
            ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self,item):

        image = self.image[item,:]
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image,(2,0,1)).astype(np.float)
        return {
            'image':torch.tensor(image,dtype=torch.float),
        }


def get_inverse_tranform():

  df = pd.read_csv('D:\courses\kaggle_dl_proj\input\data.csv')
  dict_inverse = {}
  for idx in range(len(df)):
      label_encode = df.loc[idx,'label_encoded']
      label_og = df.loc[idx,'label']
        
      if(label_encode not in dict_inverse):
          dict_inverse[label_encode] = label_og
            
  return dict_inverse
    

def predict(model,data_loader):

    for bi,d in enumerate(data_loader):
        
        with torch.no_grad():
            
            image = d['image']
            image = image.to(DEVICE,dtype=torch.float)
            output = model(image)
            _,pred = torch.max(output.data,1)
            print('Output Label is ',dict_inverse[pred.item()])

if __name__ == '__main__':

    dataset = ASIDataSetTest(
                            img_height=config.IMG_HEIGHT,
                            img_width = config.IMG_WIDTH,
                            mean = config.MODEL_MEAN,
                            std = config.MODEL_STD,
                            file_name = 'hand1_b_bot_seg_4_cropped.jpeg'
                            )

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size = config.TEST_BATCH_SIZE,
                                            shuffle=False
                                            )

    #inverse the label encoding previously computed during preprocess phase
    dict_inverse = get_inverse_tranform()

    model = MODEL_DISPATCHER[config.BASE_MODEL](pretrained=False)    
    model.load_state_dict(torch.load(r'D:\courses\kaggle_dl_proj\resnet34_FOLD_0.bin'))
    model.to(DEVICE)
    print('Calling prediction')
    predict(model,data_loader)