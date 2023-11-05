import streamlit as st
import pandas as pd
import joblib as jl
import random

happy = 'smile grinning grin smiley blush yum relieved smirk'.split()
sad = 'unamused expressionless pensive cry sob'.split()

v, model = jl.load('vectorizer'), jl.load('model')

df = pd.read_csv('yelp.csv').sample(20)

st.markdown('# Yelp Sentiment Analysis')
st.header("Created using SciKit-Learn's Random Forest Algorithm.")
st.subheader('[The Dataset](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set)')
st.markdown('**Trained with Yelp review labelled data with 1000 entries, about 80% accuracy.**')

st.header('  ')

sentence = st.text_input('Enter a Food Review')
try:
    if sentence:
        sentence_count = v.transform([sentence])
        prediction = model.predict(sentence_count)
        st.write('# Prediction:')

        if prediction == 1:
            st.write('# :' + random.choices(happy)[0] + ':')
        else:
            st.write('# :' + random.choices(sad)[0] + ':')
except:
    st.write('# An Error Occured. Please Try Later.')

st.header('  ')

st.header('Random Data Sample (Reload to get new sample)')
st.table(df)

st.header('  ')
