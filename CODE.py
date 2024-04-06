#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
real_news=pd.read_csv('True.csv')
fake_news=pd.read_csv('Fake.csv')


# In[6]:


real_news.head()


# In[7]:


fake_news.head()


# In[8]:


real_news['Isfake']=0
fake_news['Isfake']=1


# In[9]:


df=pd.concat([real_news,fake_news])


# In[10]:


df


# In[11]:


df.describe()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
#to show it in the notebook itself
# %matplotlib inline    
plt.figure(figsize=(4,4))
sns.countplot(x=df["Isfake"])
plt.show()
# plt.legend()


# In[13]:


plt.figure(figsize=(9,7))
sns.countplot(x=df["subject"])


# In[14]:


df=df[["text","Isfake"]]


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df.size


# In[18]:


df.isnull().sum()


# In[19]:


df.columns


# In[20]:


# to remove the stop words
from nltk.corpus import stopwords


# In[21]:


# to make it in string format
from nltk.stem.porter import PorterStemmer


# In[22]:


port_stem=PorterStemmer()


# In[23]:


port_stem


# In[24]:


# for regular expression 
import re,string


# In[25]:


from string import punctuation


# In[26]:


stop_words=set(stopwords.words('english'))
punctuation=list(string.punctuation)
stop_words.update(punctuation)


# In[27]:


from bs4 import BeautifulSoup


# In[28]:


def string_html(text):
    soup=BeautifulSoup(text,"html.parser")
    return soup.get_text()


# In[29]:


def remove_square_brackets(text):
    return re.sub('\[[^]]*\]','',text)


# In[30]:


def remove_URL(text):
    return re.sub(r'http\S+','',text)


# In[31]:


def remove_stopwords(text):
    final_text=[]
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)


# In[33]:


def clean_text_data(text):
    text=string_html(text)
    text=remove_square_brackets(text)
    text=remove_stopwords(text)
    text=remove_URL(text)
    return text


# In[34]:


df['text']=df['text'].apply(clean_text_data)


# In[35]:


# to remove stop regular expression 
# def stemming(content):
#     con=re.sub('[^a-zA-Z]',' ',content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con


# In[36]:


# stemming('Hi this is chando')


# In[37]:


# df['text']=df['text'].apply(stemming)
# print(df['text'])


# In[38]:


# Word Cloud - image consisting od words from our data
from wordcloud import WordCloud


# In[39]:


# For True news
plt.figure(figsize=(10,10))
wc=WordCloud(max_words=500,width=1600,height=800).generate(" ".join(df[df.Isfake==0].text))
plt.axis("off")
plt.imshow(wc,interpolation='bilinear')


# In[41]:


# For Fake news
plt.figure(figsize=(10,10))
wc=WordCloud(max_words=500,width=1600,height=800).generate(" ".join(df[df.Isfake==1].text))
plt.axis("off")
plt.imshow(wc,interpolation='bilinear')


# In[36]:


# to Differentiate the true and fake news with different image
# from PIL import Image
# thumb="path"
# icon=Image.open(thumb)
# mask=Image.new(mode="RGB",size=icon.size,color=(255,255,255))
# mask.paste(icon,box=icon)
# rgb_array=np.array(mask)


# In[37]:


# # true news
# plt.figure(figsize=(10,10))
# wc=WordCloud(mask=rgb_array,max_words=2000,width=1600,height=800)
# wc.generate(" ".join(df[df.Isfake==0].text))
# plt.axis("off")
# plt.imshow(wc,interpolation='bilinear')


# In[38]:


# skull="path"
# icon=Image.open(skull)
# mask=Image.new(mode="RGB",size=icon.size,color=(255,255,255))
# mask.paste(icon,box=icon)
# rgb_array=np.array(mask)


# In[39]:


# # fake news
# plt.figure(figsize=(10,10))
# wc=WordCloud(mask=rgb_array,max_words=2000,width=1600,height=800)
# wc.generate(" ".join(df[df.Isfake==1].text))
# plt.axis("off")
# plt.imshow(wc,interpolation='bilinear')


# In[40]:


# divide it into dependent and independent
x=df['text']
y=df['Isfake']


# In[41]:


# divide it into train and test
from sklearn.model_selection import train_test_split


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[43]:


# feed them using tfid vetorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer()


# In[44]:


# convert them to vector
x_train=vect.fit_transform(x_train)
x_test=vect.transform(x_test)


# In[45]:


x_train.shape
x_test.shape


# In[46]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report,roc_curve, auc
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[47]:


# Linear SVC
from sklearn.svm import LinearSVC
linearsvc=LinearSVC()
linearsvc.fit(x_train,y_train)
y_pred_1 = linearsvc.predict(x_test)


# In[48]:


accuracy = accuracy_score(y_test, y_pred_1)
confusion = confusion_matrix(y_test, y_pred_1)
recall = recall_score(y_test, y_pred_1)
precision = precision_score(y_test, y_pred_1)
f1 = f1_score(y_test, y_pred_1)
r_squared = r2_score(y_test, y_pred_1)
mae = mean_absolute_error(y_test, y_pred_1)
mse = mean_squared_error(y_test, y_pred_1)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)
print("R-squared score:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


# In[49]:


plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.tight_layout()
plt.show()


# In[50]:


# machine learning model - Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
y_pred_2=decision_tree.predict(x_test)


# In[51]:


accuracy = accuracy_score(y_test, y_pred_2)
confusion = confusion_matrix(y_test, y_pred_2)
recall = recall_score(y_test, y_pred_2)
precision = precision_score(y_test, y_pred_2)
f1 = f1_score(y_test, y_pred_2)
r_squared = r2_score(y_test, y_pred_2)
mae = mean_absolute_error(y_test, y_pred_2)
mse = mean_squared_error(y_test, y_pred_2)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)
print("R-squared score:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


# In[52]:


fpr, tpr, _ = roc_curve(y_test, y_pred_2)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[53]:


# Machine learning model - LogisticRegression
from sklearn.linear_model import LogisticRegression
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred_3=logistic_regression.predict(x_test)


# In[54]:


accuracy = accuracy_score(y_test, y_pred_3)
confusion = confusion_matrix(y_test, y_pred_3)
recall = recall_score(y_test, y_pred_3)
precision = precision_score(y_test, y_pred_3)
f1 = f1_score(y_test, y_pred_3)
r_squared = r2_score(y_test, y_pred_3)
mae = mean_absolute_error(y_test, y_pred_3)
mse = mean_squared_error(y_test, y_pred_3)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)
print("R-squared score:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


# In[55]:


# machine learning model - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier()
random_forest.fit(x_train,y_train)
y_pred_4=random_forest.predict(x_test)


# In[56]:


accuracy = accuracy_score(y_test, y_pred_4)
confusion = confusion_matrix(y_test, y_pred_4)
recall = recall_score(y_test, y_pred_4)
precision = precision_score(y_test, y_pred_4)
f1 = f1_score(y_test, y_pred_4)
r_squared = r2_score(y_test, y_pred_4)
mae = mean_absolute_error(y_test, y_pred_4)
mse = mean_squared_error(y_test, y_pred_4)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)
print("R-squared score:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


# In[57]:


# GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting = GradientBoostingClassifier()
gradient_boosting.fit(x_train, y_train)
y_pred_7 = gradient_boosting.predict(x_test)


# In[58]:


accuracy = accuracy_score(y_test, y_pred_7)
confusion = confusion_matrix(y_test, y_pred_7)
recall = recall_score(y_test, y_pred_7)
precision = precision_score(y_test, y_pred_7)
f1 = f1_score(y_test, y_pred_7)
r_squared = r2_score(y_test, y_pred_7)
mae = mean_absolute_error(y_test, y_pred_7)
mse = mean_squared_error(y_test, y_pred_7)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)
print("R-squared score:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)


# In[59]:


# # MLPClassifier
# from sklearn.neural_network import MLPClassifier
# neural_network = MLPClassifier()
# neural_network.fit(x_train, y_train)
# y_pred_8 = neural_network.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred_8)
# confusion = confusion_matrix(y_test, y_pred_8)
# recall = recall_score(y_test, y_pred_8)
# precision = precision_score(y_test, y_pred_8)
# f1 = f1_score(y_test, y_pred_8)
# r_squared = r2_score(y_test, y_pred_8)
# mae = mean_absolute_error(y_test, y_pred_8)
# mse = mean_squared_error(y_test, y_pred_8)
# print("Accuracy: ", accuracy)
# print("Confusion Matrix: \n", confusion)
# print("Recall: ", recall)
# print("Precision: ", precision)
# print("F1 Score: ", f1)
# print("R-squared score:", r_squared)
# print("Mean Absolute Error:", mae)
# print("Mean Squared Error:", mse)


# In[60]:


# # KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier()
# knn_classifier.fit(x_train, y_train)
# y_pred_6 = knn_classifier.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred_6)
# confusion = confusion_matrix(y_test, y_pred_6)
# recall = recall_score(y_test, y_pred_6)
# precision = precision_score(y_test, y_pred_6)
# f1 = f1_score(y_test, y_pred_6)
# r_squared = r2_score(y_test, y_pred_6)
# mae = mean_absolute_error(y_test, y_pred_6)
# mse = mean_squared_error(y_test, y_pred_6)
# print("Accuracy: ", accuracy)
# print("Confusion Matrix: \n", confusion)
# print("Recall: ", recall)
# print("Precision: ", precision)
# print("F1 Score: ", f1)
# print("R-squared score:", r_squared)
# print("Mean Absolute Error:", mae)
# print("Mean Squared Error:", mse)


# In[64]:


# # Naive Bayes Classifier
# x_train_dense = x_train.toarray()
# x_test_dense = x_test.toarray()
# from sklearn.naive_bayes import GaussianNB
# naive_bayes = GaussianNB()
# naive_bayes.fit(x_train, y_train)
# y_pred_9 = naive_bayes.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred_9)
# confusion = confusion_matrix(y_test, y_pred_9)
# recall = recall_score(y_test, y_pred_9)
# precision = precision_score(y_test, y_pred_9)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Initialize Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()

batch_size = 1000  # Set the batch size according to your available memory

# Convert and process the data in batches
for i in range(0, x_train.shape[0], batch_size):
    x_train_batch = x_train[i:i+batch_size].toarray()
    y_train_batch = y_train[i:i+batch_size]
    naive_bayes.partial_fit(x_train_batch, y_train_batch, classes=np.unique(y_train))

# Predict using the dense test data
y_pred_nb = []

for i in range(0, x_test.shape[0], batch_size):
    x_test_batch = x_test[i:i+batch_size].toarray()
    y_pred_nb.extend(naive_bayes.predict(x_test_batch))

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_nb)
confusion = confusion_matrix(y_test, y_pred_nb)

# Now you can proceed with printing accuracy, confusion matrix, etc.


# f1 = f1_score(y_test, y_pred_9)
print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", confusion)
print("Recall: ", recall)
print("Precision: ", precision)
print("F1 Score: ", f1)


# In[65]:


def output_label(n):
  if n==0:
    return "It is a True News"
  elif n==1:
    return "It is a Fake News"


# In[72]:


def manual_testing(df):
  testing_news={"text": [df]}
  new_def_test=pd.DataFrame(testing_news)
  new_def_test["text"]=new_def_test["text"].apply(clean_text_data)
  new_x_test=new_def_test["text"]
  new_xv_test=vect.transform(new_x_test)
  y_pred_7 = gradient_boosting.predict(new_xv_test)
  y_pred_4=random_forest.predict(new_xv_test)
  y_pred_3=logistic_regression.predict(new_xv_test)
  y_pred_2=decision_tree.predict(new_xv_test)
  y_pred_1 = linearsvc.predict(new_xv_test)
  return "\n\nLR Prediction: {}  \n Rf Prediction: {}  \n gb Prediction: {}  \ndt Prediction: {}  \n lsvc Prediction: {} ".format(output_label(y_pred_1[0]),output_label(y_pred_2[0]),output_label(y_pred_3[0]),output_label(y_pred_4[0]),output_label(y_pred_7[0]))


# In[73]:


news_article=str(input())


# In[74]:


manual_testing(news_article)

