import pandas as pd
import config
import matplotlib.pyplot as plt

def read_data(fold=0):

    df = pd.read_csv(config.DIR_PATH+f'/metric_fold_{fold}.csv',names=['Epoch','Train_Acc','Train_Loss','Val_Acc','Val_Loss'])
    
    df['Epoch']  = df['Epoch'].apply(lambda x: x.split(' ')[1])

    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: round(float(x.split(':')[1]),2))

    return df

def plot(x,y,z,label_x,label_y,x_axis,y_axis):

    plt.plot(z,x,'g',label = label_x)
    plt.plot(z,y,'b',label=label_y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = read_data(fold=0)
    plot(df['Train_Acc'].values,df['Val_Acc'].values,df['Epoch'].values,'Train_Acc','Val_Acc','Epochs','Accuracy')
    plot(df['Train_Loss'].values,df['Val_Loss'].values,df['Epoch'].values,'Train_Loss','Val_Loss','Epochs','Loss')
