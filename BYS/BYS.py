import numpy as np
#本次实验为朴素贝叶斯分类器，采用拉普拉斯平滑处理

def create_data():
    x_point=np.linspace(1,6,10)[:,np.newaxis]#同前两个实验，采用二位变量及标签

    noise=np.random.normal(0,1,x_point.shape)
    
    y_point=0.5*x_point+0.5+noise

    data=np.hstack((x_point,y_point))
    labels=[1,2,3,4,5,6,7,8,9,10]
    return data,labels

A=1#设置平滑值，确保不会出现概率为0



def classify(train_data,labels,input):
    #先根据频率求联合概率（根据先验概率以及条件概率求）

    #先验概率（边缘概率）字典长度与label不一定相同，只在乎每类标签的概率
    p_xianyan={}
    for i in labels:
        p_xianyan[i]=(labels.count(i)+A)/(len(labels)+A)
    #条件概率
    p_tiaojian1={}
    p_tiaojian2={}
    #先把ndarray中的二维向量转化为一维数组，
    train_data1=list(train_data[:,0])
    train_data2=list(train_data[:,1])
    for i in train_data1:
        p_tiaojian1[i]=0
    for i in train_data2:
        p_tiaojian2[i]=0
   
    #用两个二维数组来表示条件概率
    
    w1=np.linspace(0,0,len(p_xianyan))
    w1=np.tile(w1,(len(p_tiaojian1),1))
    w2=np.linspace(0,0,len(p_xianyan))
    w2=np.tile(w2,(len(p_tiaojian2),1))
  
   
  
    for x in range(len(p_tiaojian1)):
        for y in range(len(p_xianyan)):
            for i in range(len(train_data)):
                if (train_data[i][0]==list(p_tiaojian1.keys())[x]) and labels[i]==list(p_xianyan.keys())[y]:
                    w1[x][y]+=1
    for x in range(len(p_tiaojian2)):
        for y in range(len(p_xianyan)):
            for i in range(len(train_data)):
                if (train_data[i][1]==list(p_tiaojian2.keys())[x]) and labels[i]==list(p_xianyan.keys())[y]:
                    w2[x][y]+=1
    #此时w1和w2为出现的联合出现的次数
    sum1=w1.sum(axis=0)
    sum2=w2.sum(axis=0)
    
    for x in range(len(p_tiaojian1)):
        for y in range(len(p_xianyan)):
           w1[x][y]=(w1[x][y]+A)/(sum1[y]+A)    
           w2[x][y]=(w2[x][y]+A)/(sum2[y]+A)  
#到此为止就训练好了,现在对输入的数据x进行预测
    x1=input[0]
    x2=input[1]
    p_forecast=[0]*len(p_xianyan)
    
    for i in range(len(p_forecast)):
        for m in range(len(p_tiaojian1)):
           
            if x1==list(p_tiaojian1.keys())[m]:
                for n in range(len(p_tiaojian2)):
                    if x2==list(p_tiaojian2.keys())[n]:
               
                        p_forecast[i]=p_xianyan[list(p_xianyan.keys())[i]]*w1[m][i]*w2[n][i]
    max_num=0.0
    now=0
   
    
    for i in range(len(p_forecast)):

        if p_forecast[i]>max_num:    
            max_num=p_forecast[i]
            now=i
    print("属于%s类的可能性最大"% list(p_xianyan.keys())[now])

                
#本实验数据较少。
data,labels=create_data()  
x=data[5][1]
classify(data,labels,[6,x])




