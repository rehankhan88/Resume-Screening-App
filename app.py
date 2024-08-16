# import streamlit as st
# import pickle 
# import re
# import nltk
# from nltk.corpus import stopwords

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# # Loading models
# clf = pickle.load(open('clf.pkl', 'rb'))
# tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# # Define the clean_resume function
# def clean_resume(resume_text):
    # resume_text = re.sub('http\s+\s',' ',resume_text)
    # resume_text = re.sub('RT|cc',' ',resume_text)
    # resume_text = re.sub('#\s+\s',' ',resume_text)
    # resume_text = re.sub('@\s+',' ',resume_text)
    # resume_text = re.sub('[%s]' % re.escape('''!"#$%&"()*+,-./:;<=>?@[\]^_`{|}~'''),' ',resume_text)
    # resume_text = re.sub(r'[^\x00-\x7f]',' ',resume_text)
    # resume_text = re.sub('\s+',' ',resume_text)
    
#     return clean_resume

# # Web app setup
# st.image('rrrr.jpg', width=100)  # Ensure 'rrrr.jpg' is in the same directory or provide the correct path

# def main():
#     st.title('Rehan Khan Resume Screening App')
#     upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
#     if upload_file is not None:
#         try:
#             resume_bytes = upload_file.read()
#             resume_text = resume_bytes.decode('utf-8')
#         except UnicodeDecodeError:
#             resume_text = resume_bytes.decode('latin-1')
            
#         cleaned_resume = clean_resume(resume_text)
#         input_feature = tfidf.transform([cleaned_resume])  # Corrected typo 'tfidfd' to 'tfidf'
#         prediction_id = clf.predict(input_feature)[0]
#         st.write(f'Prediction ID: {prediction_id}')


# # Python main
# if __name__ == "__main__":
#     main()


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
    
    # Remove punctuation and other unwanted characters
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub('http\s+\s',' ',resume_text)
    resume_text = re.sub('RT|cc',' ',resume_text)
    resume_text = re.sub('#\s+\s',' ',resume_text)
    resume_text = re.sub('@\s+',' ',resume_text)
    resume_text = re.sub('[%s]' % re.escape('''!"#$%&"()*+,-./:;<=>?@[\]^_`{|}~'''),' ',resume_text)
    resume_text = re.sub(r'[^\x00-\x7f]',' ',resume_text)
    resume_text = re.sub('\s+',' ',resume_text)
    
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
        input_feature = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_feature)[0]
        st.write(f'Prediction ID: {prediction_id}')
        
        # Map category ID to category name
        category_mapping= {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }
        
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Predicted Category", category_name)
        



# Python main
if __name__ == "__main__":
    main()
