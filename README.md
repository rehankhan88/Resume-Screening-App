# Resume-Screening-App
Author 
<br>
Rehan Khan bajaur

git
<p>Resume-Screening-App is a machine learning–based web application that automatically analyzes resumes and predicts the most relevant job category based on the content of the resume. The goal of this project is to demonstrate how traditional NLP techniques and a lightweight classification model can be packaged into a simple, usable application for HR and recruitment workflows.

The application provides a browser-based interface where users can upload resumes in text or PDF format. The uploaded resume is cleaned, tokenized, vectorized using TF-IDF, and then passed to a trained classification model that predicts the most suitable job category.

This project focuses on practical ML implementation, not research-grade modeling.</p>


# Problem Statement

Manual resume screening is slow, inconsistent, and difficult to scale. Recruiters often need to quickly identify which job domain a resume best fits before deeper review. This application addresses that problem by:

Automatically extracting text from resumes

Cleaning and standardizing resume content

Classifying resumes into predefined job categories

It does not attempt candidate ranking or hiring decisions. It is a classification support tool only.

How the Application Works
# 1. Resume Upload

Users upload a resume in TXT or PDF format via a Streamlit web interface.

PDF files are converted to text using a PDF parser.

Text files are decoded using UTF-8 or fallback encodings.

 # 2. Text Preprocessing

The resume text goes through multiple preprocessing steps:

Lowercasing

Removal of URLs, special characters, numbers, and non-ASCII symbols

Tokenization using NLTK

Stopword removal

Whitespace normalization

These steps reduce noise and improve model consistency.

# 3. Feature Extraction

Cleaned resume text is transformed using a TF-IDF vectorizer

This converts text into numerical features representing word importance

TF-IDF was chosen because it is:

Fast

Interpretable

Suitable for smaller datasets

Easy to deploy in lightweight environments

# 4. Model Prediction

A pre-trained machine learning classifier (stored as clf.pkl) is loaded

The TF-IDF vector is passed to the model

The model outputs a predicted category ID

The category ID is mapped to a human-readable job role

 # Example categories include:

Data Science

DevOps Engineer

Python Developer

Web Designing

Business Analyst

Network Security Engineer

And others

5. Result Display

The predicted job category is displayed in the UI

A cleaned text preview is optionally shown for transparency

Technology Stack

Frontend

Streamlit (Python-based web UI)

Machine Learning

Scikit-learn

TF-IDF Vectorization

Classical ML classifier (Logistic Regression or similar)

Natural Language Processing

NLTK (tokenization and stopword removal)

File Handling

PyPDF2 (PDF text extraction)

Language

# Python 3

Project Structure
Resume-Screening-App/
│
├── app.py              # Main Streamlit application
├── clf.pkl             # Trained ML classification model
├── tfidf.pkl           # TF-IDF vectorizer
├── rrrr.jpg            # App logo (optional)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

# How to Run the Application
1. Clone the repository
git clone https://github.com/rehankhan88/Resume-Screening-App.git
cd Resume-Screening-App

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
streamlit run app.py


The application will open in your browser at:

http://localhost:8501

Model Files Requirement

The following files must exist in the project root:

clf.pkl – trained classification model

tfidf.pkl – trained TF-IDF vectorizer

Without these files, predictions cannot run.

Known Limitations

The model is trained on limited, domain-specific data

PDF text extraction may fail on scanned or image-based resumes

No model retraining pipeline is included

No bias or fairness evaluation is implemented

Not suitable for production hiring decisions

This project is intended for learning and demonstration purposes only.

# Possible Improvements

Add model training and evaluation scripts

Convert to REST API (FastAPI) for backend use

Deploy model using AWS SageMaker

Add CI/CD and MLOps pipeline

Improve PDF parsing using OCR

Introduce explainability (feature importance)

Purpose of This Project

This project demonstrates:

End-to-end ML application development

Practical NLP preprocessing

Model deployment in a simple web interface

Understanding of real-world data issues

It is designed as a portfolio and learning project, not a production hiring system.
