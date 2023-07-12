import streamlit as st
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Case Folding dan Pembersihan Teks
def casefolding(komen):
    komen = komen.lower()
    komen = komen.strip(" ")
    komen = re.sub(r'(<[A-Za-z0-9]+>)|(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',"",komen)
    return komen

# Tokenization
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

# Filtering
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

# Stemming
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

# Buka data Pickle
with open('pickle/tfIdf.pkl', mode='rb') as tf:
    tfid_load = pickle.load(tf)
with open('pickle/model.pkl', mode='rb') as model:
    model_load = pickle.load(model)
with open('pickle/clf.pkl', mode='rb') as cl:
    clf_load = pickle.load(cl)

# Streamlit
st.title('Analisis Sentimen Komentar')
comment = st.text_input('Masukan Komentar')
ok = st.button('OK')

if ok:
    dt_pred_clean = casefolding(comment)
    dt_pred_clean = token(dt_pred_clean)
    dt_pred_clean = stopword_removal(dt_pred_clean)
    dt_pred_clean = stemming(dt_pred_clean)

    dt_pred = [dt_pred_clean]
    dt_pred_tfid = tfid_load.transform(dt_pred)
    pred = clf_load.predict(dt_pred_tfid)
    if pred == "negative":
        st.error(pred[0])
    elif pred == "positive":
        st.success(pred[0])

