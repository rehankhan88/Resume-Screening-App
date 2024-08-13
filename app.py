import streamlit as st
import pickle 
import re
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Define the clean_resume function
def clean_resume(resume_text):
    # Lowercase the text
    resume_text = resume_text.lower()
    
    # Remove numbers
    resume_text = re.sub(r'\d+', '', resume_text)
    
    # Remove punctuation
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    
    # Tokenize the text
    words = nltk.word_tokenize(resume_text)
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Rejoin words into a single string
    cleaned_resume = ' '.join(words)
    
    return cleaned_resume

# Web app setup
st.image('rrrr.jpg', width=100)  # Ensure 'rrrr.jpg' is in the same directory or provide the correct path

def main():
    st.title('Rehan Khan Resume Screening App')
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            
        cleaned_resume = clean_resume(resume_text)
        input_feature = tfidf.transform([cleaned_resume])  # Corrected typo 'tfidfd' to 'tfidf'
        prediction_id = clf.predict(input_feature)[0]
        st.write(f'Prediction ID: {prediction_id}')


# Python main
if __name__ == "__main__":
    main()
