#coding:utf8
import numpy as np
import math



def normal(x,mu,sigma):
    return 1.0/(sigma*math.sqrt(math.pi*2))*math.exp(-(x-mu)*(x-mu)/2/sigma/sigma)

def e_step(data,class_data):
    rik=list()
    for i in range(len(data)):
        rik_now=list()
        all=compute_allk(data[i],class_data)
        for k in range(len(class_data)):
            rik_now.append(conpute_rik(data[i],class_data[k])/all)
        rik.append(rik_now)
    return rik

def conpute_rik(datai,class_datak):
    return class_datak[2]*normal(datai,class_datak[0],class_datak[1])
def compute_allk(datai,class_data):
    all=0.0
    for i in class_data:
        all+=conpute_rik(datai,i)
    return all

def m_step(data,class_data,rik):
    for i in range(len(rik[0])):
        classi=list()
        ni=0.0
        for j in range(len(rik)):
            ni+=rik[j][i]
        #计算ni结束



        mui=0.0
        for j in range(len(rik)):
            mui+=(rik[j][i]*data[j])
        mui/=ni
        classi.append(mui)
        #mui更新结束
        sigmai=0.0
        for j in range(len(rik)):
            sigmai+=rik[j][i]*(data[j]-class_data[i][0])*(data[j]-class_data[i][0])
        sigmai/=ni
        classi.append(sigmai)
        #sigmai更新结束
        classi.append(ni/len(rik))
        class_data[i]=classi
    return class_data

x1=np.random.normal(2,2,500)
x2=np.random.normal(5,5,200)
x3=np.random.normal(10,10,300)
data=np.hstack((x1,x2,x3))



class_data=[[1.5,1,0.45],[5.0,1.0,0.15],[11,1.0,0.4]]

for i in range(100):
    #print class_data
    rik=e_step(data,class_data)
    class_data=m_step(data,class_data,rik)
    print class_data

print class_data


