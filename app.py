import streamlit as st
import pickle 
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Web app setup
st.image('rrrr.jpg', width=100)  # Ensure 'rrrr.jpg' is in the same directory or provide the correct path

def main():
    st.title('Rehan Khan Resume Screening App')
    st.file_uploader('Upload Resume', type=['txt', 'pdf'])


#python main
if __name__ == "__main__":
    main()
