import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import csv




##Naive Bayes


def compara(origem, destino):
    soma = 0
    for i in range(len(origem)):
        parcial = origem[i] - destino[i]
        parcial = parcial*parcial
        soma += parcial
    return 1- ((soma ** 0.5)/len(origem))
        


dataset = pd.read_csv("Tweets_Mg.csv")
print('O dataset lido tem', dataset['id'].count(), 'tuplas')


tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

ngram = 0
while(True):
    resp = int(input('tecle 1 para o modelo unigram e 2 para o modelo bigram: '))
    if resp == 1 or resp == 2:
        ngram = resp
        break


if ngram==1:
    vectorizer = CountVectorizer(analyzer="word")
else:
    vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)



testes = []
gabarito = []

with open('testes.csv', 'r') as ficheiro:
    reader = csv.reader(ficheiro)
    for linha in reader:
        testes.append(linha[2])
        gabarito.append(linha[9])

freq_testes = vectorizer.transform(testes)
resp = modelo.predict(freq_testes)
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print('Acuracia do modelo: ',metrics.accuracy_score(classes,resultados))

print('\n\nMatriz de Confusao\n')
print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True),'')

nota_gabarito = []
nota_resp = []

for i in range(len(gabarito)):
    if gabarito[i] == 'Positivo':
        nota_gabarito.append(3)
    elif gabarito[i] == 'Neutro':
        nota_gabarito.append(2)
    elif gabarito[i] == 'Negativo':
        nota_gabarito.append(1)

    if resp[i] == 'Positivo':
        nota_resp.append(3)
    elif resp[i] == 'Neutro':
        nota_resp.append(2)
    elif resp[i] == 'Negativo':
        nota_resp.append(1)
		
erro_quadratico = compara(nota_gabarito, nota_resp)
print('\n\nerro quadratico: ', erro_quadratico)