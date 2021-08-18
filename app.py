import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def main():
    df = pd.read_csv('text.csv')

    chosen_emotion = st.selectbox('anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust')

    col = [chosen_emotion, 'VALUE']
    df = df[col]
    df = df[pd.notnull(df['VALUE'])]
    df.columns = [chosen_emotion, 'VALUE']

    df['category_id'] = df[chosen_emotion]
    category_id_df = df[[chosen_emotion, 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', chosen_emotion]].values)

    X_train, X_test, y_train, y_test = train_test_split(df['VALUE'], df[chosen_emotion], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = RandomForestClassifier().fit(X_train_tfidf, y_train)

    user_input = st.text_area("Type text here.",)

    if st.button("GO"):
        st.write(clf.predict(count_vect.transform([user_input])))


if __name__ == '__main__':
    main()

