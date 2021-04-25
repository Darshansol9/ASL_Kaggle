import numpy as np
import config
import pandas as pd
import joblib
import albumentations
from PIL import Image
import torch



class ASIDataSet:

    def __init__(self,img_height,img_width,folds,mean,std):

        df = pd.read_csv(config.DIR_PATH+'/data.csv')
        df = df[df.kfolds.isin(folds)].reset_index(drop=True)
        
        self.label = df['label_encoded'].values
        self.image_ids = df['image_id'].values

        if(len(folds) != 1):
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                          scale_limit=0.1,
                                          rotate_limit=5,
                                          p=0.9),
                albumentations.Normalize(mean,std,always_apply=True)
            ])

        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,item):

        image = joblib.load(config.DIR_PATH+f'/{self.image_ids[item]}.pkl')
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image,(2,0,1)).astype(np.float)
        return {
            'image':torch.tensor(image,dtype=torch.float),
            'label':torch.tensor(self.label[item],dtype=torch.long)
        }


'''
if __name__ == '__main__':

    df = get_dataframe()
    convert_to_pickle(df)

    dataset = ASIDataSet(img_height = config.IMG_HEIGHT,
                        img_width = config.IMG_WIDTH,
                        folds = config.FOLDS[0],
                        mean = config.MODEL_MEAN,
                        std = config.MODEL_STD 
                        )
    
    img,label = dataset[0]['image'],dataset[0]['label']
    npimg = img.numpy()
    print(f'Label {label}')
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
'''