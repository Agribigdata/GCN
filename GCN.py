﻿import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from modules import layers
from numpy import *
from dataset_train import set_t_12_d1_graph as settd
from torch import nn

class Net(torch.nn.Module):
    def __init__(self,input_dim=8,hidden_size=128):
        super(Net, self).__init__()
        self.conv1 = layers.GCN(input_dim, hidden_size,8)
        self.fc=nn.Linear(hidden_size,2)
    def forward(self, data):
        adj=data[0]
        adj=adj.to(DEVICE)
        x=data[1]
        x = x.to(DEVICE)
        x = self.conv1(adj,x)
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out

# 超参数定义
DEVICE = "cuda:0"
LEARNING_RATE = 1e-3  #学习率
WEIGHT_DACAY = 1e-4 #正则化系数
EPOCHS =150      #完整遍历训练集的次数
model = Net().to(DEVICE)  # 如果gpu>1 用DataParallel()包裹 单机多卡 数据并行
criterion = nn.CrossEntropyLoss().to(DEVICE)  # 多分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)  # Adam优化器
final_indice=[]
final_y=[]
def evaluate(model,loader,flag):
    global final_indice,final_y
    model.eval()
    correct_all=[]
    list_indice = []
    list_y = []
    with torch.no_grad():
        for data in loader:
            logits=model(data)
            _,indices=torch.max(logits,dim=1)
            y = data[2].to(DEVICE)
            correct=torch.sum(indices==y)
            final_corr=correct.item()*1.0/len(y)
            correct_all.append(final_corr)
            list_indice.append(indices.cpu().numpy())
            list_y.append(y.cpu().numpy())
        if flag=="test":
            final_indice+=list_indice
            final_y+=list_y
        return mean(correct_all)
def train(model,optimizer,criterion,train_data,val_data,randomstate):
    best_val_acc=0
    for epoch in range(EPOCHS):
        print('Epoch{}'.format(epoch))
        model.train()
        data_y_es=[]
        losses=[]
        logited=[]
        for data in train_data:
            #print(data.shape)
            logits=model(data)
            y = data[2].to(DEVICE)
            loss=criterion(logits,y.long())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, val_data,flag="val")
        loss_sum=round(sum(losses)/len(train_data),4)
        print('loss = {:.4f}'.format(loss_sum))
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'save_model/'+randomstate+'xxx.pth')
        print("current val acc = {}, best val acc = {}".format(acc, best_val_acc))

if __name__=="__main__":
    kfold=10
    accuracy_score_list = []
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0
    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0
    test_accuracy = 0
    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0
    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    lenset = kfold
    path = "xxxx"
    for randomstate in range(kfold):
        print(randomstate)
    #randomstate=2021
        # spilt_data(fin_path, randomstate)
        torch.manual_seed(0)
        train_data = settd.getloader(path, randomstate,"train")
        test_data = settd.getloader(path,  randomstate, "test")
        val_data = settd.getloader(path, randomstate, "val")
        train(model,optimizer,criterion,train_data,val_data,str(randomstate))
        model.load_state_dict(torch.load('save_model/'+str(randomstate)+'xxx.pth'))
        evaluate(model,test_data,flag="test")
        #model.zero_grad()
        #optimizer.zero_grad()
        model = Net().to(DEVICE)  # 如果gpu>1 用DataParallel()包裹 单机多卡 数据并行
        criterion = nn.CrossEntropyLoss().to(DEVICE)  # 多分类交叉熵损失
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)  # Adam优化器
        y_final=np.concatenate(final_y)
        indice_final=np.concatenate(final_indice)
        accuracy_score_list.append(classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy'])
        test_road_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['0.0']['precision']
        test_road_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['0.0']['recall']
        test_road_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['0.0']['f1-score']
        test_field_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['1.0']['precision']
        test_field_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['1.0']['recall']
        test_field_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['1.0']['f1-score']
        test_accuracy += classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy']
        test_macro_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg'][
            'precision']
        test_macro_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['recall']
        test_macro_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg'][
            'f1-score']
        test_weight_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'precision']
        test_weight_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'recall']
        test_weight_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg'][
            'f1-score']

        final_indice = []
        final_y = []
    print(accuracy_score_list)
    print('               precision    recall    f1score')
    print('        0    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('        1    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))
#



