import streamlit as st
import pickle

with open('pickle/tfIdf.pkl', mode='rb') as tf:
    tfid_load = pickle.load(tf)
with open('pickle/model.pkl', mode='rb') as model:
    model_load = pickle.load(model)
with open('pickle/clf.pkl', mode='rb') as cl:
    clf_load = pickle.load(cl)

st.title('Analisis Sentimen Komentar')
comment = st.text_input('Masukan Komentar')
ok = st.button('OK')

if ok:
    dt_pred = [comment]
    dt_pred_tfid = tfid_load.transform(dt_pred)
    pred = clf_load.predict(dt_pred_tfid)
    if pred == "negative":
        st.error(pred[0])
    elif pred == "positive":
        st.success(pred[0])