# -*- coding: utf-8 -*-
"""
Created on Fri Jan 7 14:22:17 2022

@author: Vladimir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv("recipes.csv")

df.shape

prvi = df.head()

df.info()

opis = df.describe()

tipovi = df.dtypes

obelezja = df.columns


#%%
df.drop('Unnamed: 0', inplace= True, axis = 1)

#provera da li ima nedostajućih vrednosti
print(df.isnull().sum().sum())

countryGroupBy = df.groupby("country").sum()

#prikaz država iz kojih su recepti
drzave = df.iloc[:,-1].unique()

#%%

#zastupljenost recepatav po drzavama
lista=[]
d1 =np.sum(df['country'] == 'southern_us') 
lista.append(d1)

d2 =np.sum(df['country'] == 'french') 
lista.append(d2)

d3 =np.sum(df['country'] == 'greek') 
lista.append(d3)

d4 =np.sum(df['country'] == 'mexican') 
lista.append(d4)

d5 =np.sum(df['country'] == 'italian') 
lista.append(d5)

d6 =np.sum(df['country'] == 'japanese') 
lista.append(d6)

d7 =np.sum(df['country'] == 'chinese') 
lista.append(d7)

d8 =np.sum(df['country'] == 'thai') 
lista.append(d8)

d9 =np.sum(df['country'] == 'british') 
lista.append(d9)

print(np.sum(df['country'] == 'southern_us'))
print(np.sum(df['country'] == 'french'))
print(np.sum(df['country'] == 'greek'))
print(np.sum(df['country'] == 'mexican'))
print(np.sum(df['country'] == 'italian'))
print(np.sum(df['country'] == 'japanese'))
print(np.sum(df['country'] == 'chinese'))
print(np.sum(df['country'] == 'thai'))
print(np.sum(df['country'] == 'british'))

#zastupljenost recepata po drzavama

plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(drzave,lista)
plt.xticks(rotation=90)
plt.show()  
#%%

# najzastupljeniji sastojci u drzavama
southern_us = df.loc[df['country']=='southern_us'].drop(['country'], axis=1)
southern_us_sum = southern_us.sum(axis=0).sort_values(ascending=False)
southern_us_percent= (southern_us.sum(axis=0)/southern_us.shape[0])

l=[]
k = 0
for i in southern_us_percent:
    if i < 0.2:
        l.append(k)
    k = k + 1
southern_us = southern_us.drop(southern_us.columns[l], axis=1)
southern_us_percent = southern_us_percent.drop(southern_us_percent.index[l])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(southern_us.columns,southern_us_percent)
plt.xticks(rotation=90)
plt.show() 

french = df.loc[df['country']=='french'].drop(['country'], axis=1)
french_sum = french.sum(axis=0).sort_values(ascending=False)
french_percent= (french.sum(axis=0)/french.shape[0])

lista=[]
k = 0
for i in french_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
french = french.drop(french.columns[lista], axis=1)
french_percent = french_percent.drop(french_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(french.columns,french_percent)
plt.xticks(rotation=90)
plt.show() 

greek = df.loc[df['country']=='greek'].drop(['country'], axis=1)
greek_sum  = greek.sum(axis=0).sort_values(ascending=False)
greek_percent= (greek.sum(axis=0)/greek.shape[0])

greek = df.loc[df['country']=='greek'].drop(['country'], axis=1)
greek_sum  = greek.sum(axis=0).sort_values(ascending=False)
greek_percent= (greek.sum(axis=0)/greek.shape[0])

lista=[]
k = 0
for i in greek_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
greek = greek.drop(greek.columns[lista], axis=1)
greek_percent = greek_percent.drop(greek_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(greek.columns,greek_percent)
plt.xticks(rotation=90)
plt.show() 

mexican = df.loc[df['country']=='mexican'].drop(['country'], axis=1)
mexican_sum = mexican.sum(axis=0).sort_values(ascending=False)
mexican_percent= (mexican.sum(axis=0)/mexican.shape[0])

lista=[]
k = 0
for i in mexican_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
mexican = mexican.drop(mexican.columns[lista], axis=1)
mexican_percent = mexican_percent.drop(mexican_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(mexican.columns,mexican_percent)
plt.xticks(rotation=90)
plt.show() 

italian = df.loc[df['country']=='italian'].drop(['country'], axis=1)
italian_sum = italian.sum(axis=0).sort_values(ascending=False)
italian_percent= (italian.sum(axis=0)/italian.shape[0])

lista=[]
k = 0
for i in italian_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
italian = italian.drop(italian.columns[lista], axis=1)
italian_percent = italian_percent.drop(italian_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(italian.columns,italian_percent)
plt.xticks(rotation=90)
plt.show() 

japanese = df.loc[df['country']=='japanese'].drop(['country'], axis=1)
japanese_sum = japanese.sum(axis=0).sort_values(ascending=False)
japanese_percent= (japanese.sum(axis=0)/japanese.shape[0])

lista=[]
k = 0
for i in japanese_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
japanese = japanese.drop(japanese.columns[lista], axis=1)
japanese_percent = japanese_percent.drop(japanese_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(japanese.columns,japanese_percent)
plt.xticks(rotation=90)
plt.show() 
 
chinese = df.loc[df['country']=='chinese'].drop(['country'], axis=1)
chinese_sum = chinese.sum(axis=0).sort_values(ascending=False)
chinese_percent= (chinese.sum(axis=0)/chinese.shape[0])

lista=[]
k = 0
for i in chinese_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
chinese = chinese.drop(chinese.columns[lista], axis=1)
chinese_percent = chinese_percent.drop(chinese_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(chinese.columns,chinese_percent)
plt.xticks(rotation=90)
plt.show() 

thai = df.loc[df['country']=='thai'].drop(['country'], axis=1)
thai_sum = thai.sum(axis=0).sort_values(ascending=False)
thai_percent= (thai.sum(axis=0)/thai.shape[0])

lista=[]
k = 0
for i in thai_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
thai = thai.drop(thai.columns[lista], axis=1)
thai_percent = thai_percent.drop(thai_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(thai.columns,thai_percent)
plt.xticks(rotation=90)
plt.show() 

british = df.loc[df['country']=='british'].drop(['country'], axis=1)
british_sum = british.sum(axis=0).sort_values(ascending=False)
british_percent= (british.sum(axis=0)/british.shape[0])

lista=[]
k = 0
for i in british_percent:
    if i < 0.2:
        lista.append(k)
    k = k + 1
british = british.drop(british.columns[lista], axis=1)
british_percent = british_percent.drop(british_percent.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(british.columns,british_percent)
plt.xticks(rotation=90)
plt.show() 

#%%
def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg


#%%knn

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(X.shape)
labels_y = y.unique()
print(labels_y)

X.describe()

y.groupby(by=y).count()

# podela podataka na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, stratify=y)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for k in [1,5,10]:
    for m in ['jaccard', 'dice']:
        indexes = kf.split(X, y)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in indexes:
            classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
            classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
            y_pred = classifier.predict(X.iloc[test_index,:])
            acc_tmp.append(recall_score(y.iloc[test_index], y_pred, average='micro'))
            fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
        print('za parametre k=', k, ' i m=', m, ' osetljivost je: ', np.mean(acc_tmp), ' a mat. konf. je:')
        #print(fin_conf_mat)

        disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
        disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
        plt.show()
        
        acc.append(np.mean(acc_tmp))
print('najbolja osetljivost je u iteraciji broj: ', np.argmax(acc))

print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))


classifier = KNeighborsClassifier(n_neighbors=10, metric='jaccard')
classifier.fit(X, y)
y_pred = classifier.predict( X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=labels_y)
#print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(y_test, y_pred))
print('preciznost mikro: ', precision_score(y_test, y_pred, average='micro'))
print('preciznost makro: ', precision_score(y_test, y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(y_test, y_pred, average='micro'))
print('osetljivost makro: ', recall_score(y_test, y_pred, average='macro'))
print('f mera mikro: ', f1_score(y_test, y_pred, average='micro'))
print('f mera makro: ', f1_score(y_test, y_pred, average='macro'))
print(labels_y)

#%%
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
acc = []
for c in [1, 10, 100]:
    for F in ['linear', 'rbf']:
        for mc in ['ovo', 'ovr']:
            indexes = kf.split(X, y)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc)
                classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
                y_pred = classifier.predict(X.iloc[test_index,:])
                acc_tmp.append(recall_score(y.iloc[test_index], y_pred, average='micro'))
                fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' osetljivost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
            #print(fin_conf_mat)

            disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
            disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
            plt.show()

            acc.append(np.mean(acc_tmp))
print('najbolja osetljivost je u iteraciji broj: ', np.argmax(acc))

print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, y.unique()))

classifier = SVC(C=1, kernel='rbf', decision_function_shape='ovo')
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=labels_y)

#print(conf_mat)
disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
plt.show()

print('procenat pogodjenih uzoraka: ', accuracy_score(y_test, y_pred))
print('preciznost mikro: ', precision_score(y_test, y_pred, average='micro'))
print('preciznost makro: ', precision_score(y_test, y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(y_test, y_pred, average='micro'))
print('osetljivost makro: ', recall_score(y_test, y_pred, average='macro'))
print('f mera mikro: ', f1_score(y_test, y_pred, average='micro'))
print('f mera makro: ', f1_score(y_test, y_pred, average='macro'))


#%%

#zastupljenost recepatav po drzavama na train skupu

print('southern_us: ', np.sum(y_train== 'southern_us'), 'french: ', np.sum(y_train == 'french'), 
      'greek: ', np.sum(y_train== 'greek'), 'mexican: ', np.sum(y_train == 'mexican'), 'italian: ', np.sum(y_train == 'italian'),
       'japanese: ', np.sum(y_train == 'japanese'), 'chinese: ', np.sum(y_train == 'chinese'), 'thai: ', np.sum(y_train == 'thai')
       , 'british: ', np.sum(y_train == 'british'))

#zastupljenost recepatav po drzavama na test skupu

print('southern_us: ', np.sum(y_test == 'southern_us'), 'french: ', np.sum(y_test == 'french'), 
      'greek: ', np.sum(y_test == 'greek'), 'mexican: ', np.sum(y_test == 'mexican'), 'italian: ', np.sum(y_test == 'italian'),
       'japanese: ', np.sum(y_test == 'japanese'), 'chinese: ', np.sum(y_test == 'chinese'), 'thai: ', np.sum(y_test == 'thai')
       , 'british: ', np.sum(y_test == 'british'))