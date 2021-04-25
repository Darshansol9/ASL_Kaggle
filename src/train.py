import config
import models
from dispacter import MODEL_DISPATCHER
from dataset import ASIDataSet
import torch.nn as nn
import torch
from tqdm import tqdm
import torch
import json
import matplotlib.pyplot as plt


DEVICE = config.DEVICE


def plot(image,output,target):


    count = 0
    for i in range(image.shape[0]):

        img = image[i].permute(1, 2, 0)
        img = img.cpu().detach().numpy()
        op = output[i].cpu().detach().numpy()
        tg = target[i].cpu().detach().numpy()

        print(f'Predicted label {op} and Actual label {tg}')
        plt.imshow(img)
        plt.show()
        if(count > 10):
            break
        
        count +=1


def loss_fn(output,target):

    return nn.CrossEntropyLoss()(output,target)

def evaluate(dataset,data_loader,model):

    model.eval()
    
    final_loss = 0
    counter = 0
    correct_val = 0


    for _,d in tqdm(enumerate(data_loader),total=int(len(dataset) / data_loader.batch_size)):
        
        with torch.no_grad():

            image = d['image']
            label = d['label']

            image = image.to(DEVICE,dtype=torch.float)
            label = label.to(DEVICE,dtype=torch.long)

            output = model(image)
            loss = loss_fn(output,label)
            
            final_loss += loss

            _,pred = torch.max(output.data,1)
            correct_val += pred.eq(label.data).sum().item()
            
            counter +=1
        
    return correct_val / len(dataset) , final_loss / counter


def train(dataset,data_loader,model,optimizer):


    model.train()

    train_loss = 0
    correct_train = 0
    counter = 0

    for bi,d in tqdm(enumerate(data_loader),total=int(len(dataset) / data_loader.batch_size)):
        
        torch.cuda.empty_cache()

        image = d['image']
        label = d['label']

        image = image.to(DEVICE,dtype=torch.float)
        label = label.to(DEVICE,dtype=torch.long)

    
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output,label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        #accuracy
        _,pred = torch.max(output.data,1)
        each_iter = pred.eq(label.data).sum().item()
        correct_train += pred.eq(label.data).sum().item()
        #print(correct_train,each_iter)
        counter +=1


    return correct_train / len(dataset), train_loss / counter



def main():

    for val_fold,train_fold in config.FOLDS.items():

        torch.cuda.empty_cache()

        model = MODEL_DISPATCHER[config.BASE_MODEL](pretrained=True)
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            patience=5,
                                                            factor=0.3,
                                                            verbose=True
                                                            )
        train_dataset = ASIDataSet(
                        img_height=config.IMG_HEIGHT,
                        img_width = config.IMG_WIDTH,
                        folds = train_fold,
                        mean = config.MODEL_MEAN,
                        std = config.MODEL_STD
                        )

        train_dataloader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=config.TRAIN_BATCH_SIZE,
                            shuffle=True,
                            num_workers=4
                            )

        valid_dataset = ASIDataSet(
                        img_height=config.IMG_HEIGHT,
                        img_width = config.IMG_WIDTH,
                        folds = [val_fold],
                        mean = config.MODEL_MEAN,
                        std = config.MODEL_STD
                        )

        valid_dataloader =torch.utils.data.DataLoader(
                            valid_dataset,
                            shuffle=True,
                            num_workers=4,
                            batch_size=config.VALID_BATCH_SIZE
                            )
        
        #print(f'Train dataset length ',len(train_dataset))
        #print(f'Valid dataset length ',len(valid_dataset))

        prev_score = float('inf')
        f = open(f'{config.DIR_PATH}/metric_fold_{val_fold}.csv','w')

        loss_increased = 0

        for epoch in range(config.EPOCHS):

            torch.cuda.empty_cache()
            
            train_acc,train_loss = train(train_dataset,train_dataloader,model,optimizer)
            val_acc,val_loss = evaluate(valid_dataset,valid_dataloader,model)
            scheduler.step(val_loss)
            
            to_print = f'Epoch {epoch}, train_acc: {train_acc}, train_loss: {train_loss}, val_acc: {val_acc}, val_loss: {val_loss}'
            f.write(to_print+'\n')
            print(to_print)

            if(val_loss < prev_score):
                prev_score = val_loss
                torch.save(model.state_dict(),f'{config.BASE_MODEL}_FOLD_{val_fold}.bin')
                loss_increased = 0
            else:
                loss_increased +=1
            
            if(loss_increased >= 2):
                print('Early Stopping executed')
                break

            #print(f'Memory allocated ',torch.cuda.memory_allocated(0))

        f.close()
        break

if __name__ == '__main__':
    main()
