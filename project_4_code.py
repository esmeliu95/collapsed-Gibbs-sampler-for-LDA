
# coding: utf-8

# In[134]:


import numpy as np
import pandas as pd
from numpy.linalg import inv
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.linalg import inv    
import random
from numpy import linalg as LA


# In[135]:


N_iter = 100


# In[136]:


from collections import Counter
tmp = []
with open('1') as f:
    for line in f:
        for word in line.split():
            tmp.append(word)
            
V = Counter(tmp)

for i in range(2,201):
    
    i = str(i)
    tmp = []
    with open(i) as f:
        for line in f:
            for word in line.split():
                tmp.append(word)

    V += Counter(tmp)

print(len(V))


# In[137]:


#load data
w = []
d = []
z = []
C_d = np.zeros((200,20))
C_t = np.zeros((20,len(V)))
look_up = {}

look_up = {}
for nde,i in enumerate(V):
    look_up[i] = nde


for i in range(1,201):
    
    index = i
    i = str(i)
    
    temp = []
    with open(i) as f:
        for line in f:
            for word in line.split():
                temp.append(word)    
    
    temp = np.array(temp)
    
    temp = np.random.permutation(temp)
    for word in temp:
        w.append(word)
        d.append(index-1)
        this_z = np.random.randint(1,21)
        z.append(this_z-1)
        
        C_d[index-1][this_z-1]+=1
        C_t[this_z-1][look_up[word]]+=1


# In[138]:


N_words = int(np.sum(C_d))


# In[139]:


P = []
K = 20
alpha = 0.1
beta = 0.01
VV = len(V)
for i in range(K):
    P.append(0)


# In[140]:


def find_bin(number,array):
    
    
    temp = np.cumsum(array)
    
    for i in range(len(temp)):
        if temp[i] >= number:
            number = i
            break
        
    
    return number


# In[141]:


N_iter = 100
for i in range(N_iter):
    for n in range(N_words):
        
        word = w[n]
        topic = z[n]
        doc = d[n]
             
        
        C_d[doc][topic] -=1
        C_t[topic][look_up[word]]-=1
        
        
            
        for k in range(K):
            left = (C_t[k][look_up[word]]+beta)/(VV*beta + np.sum(C_t[k]))
            right = (C_d[doc][k]+alpha)/(K*alpha+np.sum(C_d[doc]))
            P[k] = left*right
        
        sum_ = np.sum(P)
        for k in range(K):
            P[k] = P[k]/sum_
        
        #print(P)
        draw = np.random.uniform(0,1)
        topic = find_bin(draw,P)
        z[n] = topic
        C_d[doc][topic] +=1
        C_t[topic][look_up[word]]+=1
        


# In[142]:


inverted_dict = {value: key for key, value in look_up.items()}


# In[143]:


CC = C_t.copy()


# In[144]:


def accuracy(y,y_):
    
    temp = 0
    
    for i in range(len(y)):
        
        k = 0
        if y_[i] >= 0.5:
            k = 1
            
        if y[i] == k:
            temp+=1
    
    return 1-(temp/len(y))


# In[145]:


top_5 = []

for i in range(len(CC)):
    
    each_row = {}

    for ii in range(CC.shape[1]):
        each_row[inverted_dict[ii]] = CC[i][ii]
    
    tmp = sorted(each_row.items(), key=lambda kv: kv[1],reverse=True)[:5]
    top_5.append(tmp)
    


# In[133]:


#df = pd.read_csv("topicwords.csv")
df = pd.DataFrame()

for i in range(len(top_5[0])):
    
    tmp_ = []
    for j in range(len(top_5)):
       
        tmp_.append(top_5[j][i][0])
        
    df[str(i)] = tmp_
    df.to_csv('topicwords.csv', index=False, header=False)


# In[146]:


topic_present = []
for i in range(len(C_d)): #for each doc
    
    tmp = []
    for j in range(len(C_d[i])): #each topic
        
        doc = i
        k = j
        
        
        temp = (C_d[doc][k]+alpha)/(K*alpha + np.sum(C_d[doc]))
        tmp.append(temp)
    topic_present.append(tmp)
        
#print(np.array(topic_present).shape)


# In[147]:


#bag of words
bag_of_words = np.zeros((200,len(V)))

for index in range(1,201):
    tmp = []
    index_number = index
    index = str(index)
    with open(index) as f:
        for line in f:
            for word in line.split():
                tmp.append(word)

    counter_for_each_doc = Counter(tmp)

    for i in range(len(bag_of_words[index_number-1])):

        if inverted_dict[i] in tmp:
            bag_of_words[index_number-1][i] = (counter_for_each_doc[inverted_dict[i]])/len(tmp)



# In[148]:


labels = np.array(pd.read_csv('index.csv',header=None))[:,1]
labels = np.vstack(labels)


# In[149]:


def split(data_set):
    
    #shuffle the data
    data_set = np.random.permutation(data_set)
    train_index = int(len(data_set)*0.6)
    train_set = data_set[:train_index]
    test_set = data_set[train_index:] 
    
    return train_set,test_set


# In[121]:


#Task 2
#Implement model selection for Bayesian logistic regression


class BLR(object):
    
    def __init__(self,X,a_ = 0.1):
        self.weight = np.zeros((1,len(X[0])+1)).T
        self.alpha = a_
        self.mu_a = 0
        self.S_N = 0
        self.theta_sqr = 0
        self.R = 0
        
    #compute sigmoid
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self,XXX,yyy):
        
        
        X = XXX.copy()
        y = yyy.copy()
        #stack one more column to X
        ones = np.ones((len(X),1))
        X = np.hstack((X,ones))
        #self.R = np.zeros((len(X),len(X)))
        
        for l in range(10):

            self.R = np.zeros((len(X),len(X)))
            theta_ = np.zeros((1,len(X)))
            self.I = np.identity(len(X[0]))    

            
            for k in range(100):
                #calculating new weight   
                for i in range(len(X)):
                    for j in range(len(X)):
                        if i == j:
                            #diagnol
                            XX = np.vstack(X[i]).T
                            
                            
                            self.R[i][j] = self.__sigmoid(XX@self.weight)[0][0]*(1-(self.__sigmoid(XX@self.weight)))[0][0]
                            #print('R shape ',self.R[i][j].shape)
                            theta_[0][i] = self.__sigmoid(XX@self.weight)[0][0]

                theta_ = theta_.T
                new_weight = self.weight - (inv((X.T@self.R@X)+ self.alpha * self.I)@(X.T@(theta_ - y)+ self.alpha*self.weight))


                diff_deno = 0 
                diff_nume = 0 

                for each_ in range(len(self.weight)):

                    diff_nume += (new_weight[each_] - self.weight[each_])**2
                    diff_deno += self.weight[each_]**2 ############################check equation

                if (np.sqrt(diff_nume))/(np.sqrt(diff_deno)) < 0.001:
                    break

                self.weight = new_weight
                theta_ = theta_.T
                
            

            matrix = X.T@self.R@X
            eigenvalues, eigenvectors = LA.eig(matrix)
            ita = 0
            

            for each_eigen_value in eigenvalues:
                ita = ita + (each_eigen_value/(self.alpha + each_eigen_value))
            
            k = self.weight.T@self.weight
            
            self.alpha = (ita/((self.weight).T@(self.weight)))


            


    def predict(self,XXX,XXX_star):
        
        X = XXX.copy()
        X_star = XXX_star.copy()
        #stack one more column to X
        ones = np.ones((len(X),1))
        X = np.hstack((X,ones))
        
        ones = np.ones((len(X_star),1))
        X_star = np.hstack((X_star,ones))
        
        
        y_pred = np.zeros((1,len(X_star))).T
        for i in range(len(X_star)):
            
            mu_a = self.weight.T @ X_star[i].T
            self.S_N = inv((X.T@self.R@X) + self.alpha*self.I)
            theta_sqr = X_star[i] @self.S_N @X_star[i].T
            y_pred[i] = mu_a /np.sqrt(1+((np.pi/8)*self.theta_sqr))
            

        return self.__sigmoid(y_pred)
    
    def return_weight(self):
        return self.weight
    
    def return_alpha(self):
        return self.alpha
            


# In[122]:


#stack
bag_of_words = np.hstack((bag_of_words,labels))
topic_present = np.hstack((topic_present,labels))

data_set_size = 10
each_incre = int(200/10)

bag_err = []
topic_err = []
topic_var = []
bag_var = []

for count_t in range(10):

    
    data_set_size+=each_incre
    tmp_bag_mean = []
    tmp_topic_mean = []
    
    
    for counts in range(30):#(30):

        bag_train, bag_test = split(bag_of_words[:data_set_size])
        topic_train,topic_test = split(topic_present[:data_set_size])

        bag_train_x = bag_train[:,:len(bag_train[0])-1]
        bag_train_y = bag_train[:,len(bag_train[0])-1]
        bag_train_y = np.vstack(bag_train_y)

        bag_test_x = bag_test[:,:len(bag_test[0])-1]
        bag_test_y = bag_test[:,len(bag_test[0])-1]
        bag_test_y = np.vstack(bag_test_y)   

        topic_train_x = topic_train[:,:len(topic_train[0])-1]
        topic_train_y = topic_train[:,len(topic_train[0])-1]
        topic_train_y = np.vstack(topic_train_y)

        topic_test_x = topic_test[:,:len(topic_test[0])-1]
        topic_test_y = topic_test[:,len(topic_test[0])-1]
        topic_test_y = np.vstack(topic_test_y)      

        lol_b = BLR(bag_train_x)
        lol_b.fit(bag_train_x,bag_train_y)
        ans_ = lol_b.predict(bag_train_x,bag_test_x)
        tmp_bag_mean.append(accuracy(bag_test_y,ans_)) 


        lol_b = BLR(topic_train_x)
        lol_b.fit(topic_train_x,topic_train_y)
        ans_A_b = lol_b.predict(topic_train_x,topic_test_x)
        tmp_topic_mean.append(accuracy(topic_test_y,ans_A_b))

    bag_err.append(np.mean(tmp_bag_mean))
    topic_err.append(np.mean(tmp_topic_mean))
    bag_var.append(np.sqrt(np.var(tmp_bag_mean)))
    topic_var.append(np.sqrt(np.var(tmp_topic_mean)))


# In[87]:


y_axis = []
for i in range(10):
    y_axis.append(i)
    
plt.errorbar(y_axis, bag_err,yerr=bag_var,label='bag of words',fmt='-o')
plt.errorbar(y_axis, topic_err,yerr=topic_var,label='LDA',fmt='-o') 
plt.legend(loc='best')
plt.ylabel('Error rate')
plt.xlabel('Trial No.')
plt.show()

