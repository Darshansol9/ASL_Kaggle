import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn import model_selection
from sklearn import preprocessing
import glob
import os
import joblib
import config
import pandas as pd
import numpy as np


def get_dataframe():
    
    dict_image = defaultdict(dict)

    for fol in os.listdir(config.DATA_PATH):
        path_to_file = os.path.join(config.DATA_PATH,fol)
        for files in os.listdir(path_to_file):
            dict_image[files] = {'file_path':os.path.join(path_to_file,files),'label':fol}

    df = pd.DataFrame.from_dict(dict_image,orient='index')
    df = df.reset_index(drop=False).rename(columns={'index':'image_id'})
    df['kfolds'] = -1

    le = preprocessing.LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'].values)
    
    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(_,val_idx) in enumerate(kf.split(X=df,y=df.label.values)):
        df.loc[val_idx,'kfolds'] = fold

    df.to_csv(config.DIR_PATH+'/data.csv',index=False)
    
    return df

def convert_to_pickle(df):

    image_ids = df['image_id'].values
    file_paths = df['file_path'].values

    for _,(img_id,file_pth) in tqdm(enumerate(zip(image_ids,file_paths)),total=len(df)):
        img  = mpimg.imread(file_pth)
        joblib.dump(img,config.DIR_PATH+f'/{img_id}.pkl')


if __name__ == '__main__':
    df = get_dataframe()
    convert_to_pickle(df)
    print('Completed')