import pickle 
import streamlit as st

classifier_in=open("classifier.pkl","rb")
clf=pickle.load(classifier_in)
def predict_banknote(variance,skewness,kurtosis,entropy):
    pred=clf.predict([[variance,skewness,kurtosis,entropy]])
    if(pred[0]>0.5):
        pred="Its a fake note"
    else:
        pred="It's a real banknote"

    return pred

variance=st.number_input("Enter the variance")
skewness=st.number_input("Enter the skewness")
kurtosis=st.number_input("Enter the kurtosis")
entropy=st.number_input("Enter the entropy")
if(st.button("Predict")):
    result=predict_banknote(variance,skewness,kurtosis,entropy)
    st.success(result)