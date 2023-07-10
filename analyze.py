import pandas as pd
data = pd.read_csv("data/dataset.csv")

#Case Folding
import re 
def casefolding(komen):
    komen = komen.lower()
    komen = komen.strip(" ")
    komen = re.sub(r'(<[A-Za-z0-9]+>)|(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',"",komen)
    return komen
data['Komen'] = data['Komen'].apply(casefolding)
print("Case Folding [Ok]")

#Tokenization
def token(komen):
    nstr = komen.split(" ")
    dat = []
    a = -1

    for i in nstr:
        a = a + 1

    if i == "":
        dat.append(a)

    p = 0
    b = 0

    for q in dat:
        b = q - p
        del nstr[b]
        p = p + 1

    return nstr

data["Komen"] = data["Komen"].apply(token)
print("Tokenization [Ok]")

#Filtering
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stopword_removal(komen):
    filtering = stopwords.words("indonesian")
    x = []
    data = []
    def myFunc(x):
        if x in filtering:
            return False
        else:
            return True
        
    fit = filter(myFunc, komen)
    for x in fit:
        data.append(x)

    return data

data["Komen"] = data["Komen"].apply(stopword_removal)
print("Filtering [Ok]")

#Stemming
print("Stemming process...")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def stemming(komen):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    kt = []

    for w in komen:
        dt = stemmer.stem(w)
        kt.append(dt)

    dt_clean = []
    dt_clean = " ".join(kt)
    return dt_clean

data["Komen"] = data["Komen"].apply(stemming)

data.to_csv("data/data_clean.csv", index=False)
data_clean = pd.read_csv("data/data_clean.csv")
print("Stemming [Ok]")

data_clean = data_clean.astype({'Komen' : 'string'})
data_clean = data_clean.astype({'Sentimen' : 'category'})
data_clean.dtypes

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfIdf = TfidfVectorizer()
data_tfIdf = tfIdf.fit_transform(data_clean["Komen"].astype("U"))
print("TF-IDF Shape : ",data_tfIdf.shape)
print("TF-IDF [Ok]")

#Splitting
from sklearn.model_selection import train_test_split

x = data_tfIdf
y = data_clean["Sentimen"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nX_train : ", x_train.shape)
print("X_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)
print("Spliting [Ok]")

#Impementasi NB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf = MultinomialNB().fit(x_train, y_train)
predicted = clf.predict(x_test)
print("\n")
print("MultinomialNB Accuracy : ", accuracy_score(y_test,predicted))
print("MultinomialNB Preccision : ", precision_score(y_test, predicted, average="binary", pos_label="negative"))
print("MultinomialNB Recall : ", recall_score(y_test, predicted, average="binary", pos_label="negative"))
print("MultinomialNB F1 Score : ", f1_score(y_test, predicted, average="binary", pos_label="negative"))
print(f"Confusion Matrix : \n {confusion_matrix(y_test, predicted)}")
print("======================================================")
print(classification_report(y_test, predicted, zero_division=0))


import streamlit as st
st.title('Analisis Sentimen Komentar')
comment = st.text_input('Masukan Komentar')
ok = st.button('OK')

if ok:
    dt_pred = [comment]
    dt_pred_tfid = tfIdf.transform(dt_pred)
    my_pred = clf.predict(dt_pred_tfid)
    if my_pred[0]=="negative":
        st.error("Negatif")
    elif my_pred[0]=="positive":
        st.success("Positif")
    else:
        st.error("yahh")
