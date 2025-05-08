# Install necessary libraries
# !pip install scikit-learn PyPDF2 pillow easyocr

import pickle
import PyPDF2
from PIL import Image
import easyocr

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the fake news detection model
with open('fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_fake_news(text):
    """
    Predicts whether the given text is fake news or not.

    Args:
        text (str): The input text.

    Returns:
        str: The prediction label ('fake' or 'not fake').
    """
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_image(image_path):
    """Extracts text from an image using EasyOCR."""
    reader = easyocr.Reader(['en'])  # Use English language
    results = reader.readtext(image_path, detail=0)
    text = " ".join(results)
    return text

# Example usage
pdf_path = "./example.pdf"
image_path = "./example_image.png"

# Extract text
pdf_text = extract_text_from_pdf(pdf_path)
image_text = extract_text_from_image(image_path)

# Predictions
pdf_prediction = predict_fake_news(pdf_text)
image_prediction = predict_fake_news(image_text)

# Output
if pdf_prediction == "fake" and image_prediction == "fake":
    print("Both PDF and image article was fake.")
elif pdf_prediction == "fake" and image_prediction == "not fake":
    print("PDF article was fake but image article was correct.")
elif pdf_prediction == "not fake" and image_prediction == "fake":
    print("Image article was fake but PDF article was correct.")
else:
    print("Both PDF and image article are correct.")