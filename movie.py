### Mood2Movie — Enhanced Version (Up to Feature 4)
# Includes: Transformer-based Emotion Detection, Better Filtering, OMDb API Poster Fetch, Multilingual Input
# Commits to be made
# Commit made at 

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from langdetect import detect
from googletrans import Translator

# Load movie dataset
df = pd.read_csv(r"E:\hackathon\movie_dataset.csv")

# Load transformer model for emotion detection
MODEL = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Translator for multilingual input
translator = Translator()

# Emotion-to-Genre Mapping
genre_filter = {
    'happy': ['Comedy', 'Romance', 'Musical'],
    'sad': ['Drama', 'Romance', 'Biography'],
    'lonely': ['Drama', 'Romance', 'Adventure'],
    'fear': ['Comedy', 'Sci-Fi', 'Mystery'],
    'surprise': ['Mystery', 'Thriller', 'Horror']
}

# Normalize model emotion labels to expected ones
emotion_alias = {
    'joy': 'happy',
    'happiness': 'happy',
    'anger': 'fear',
    'sadness': 'sad',
    'fear': 'fear',
    'surprise': 'surprise',
    'lonely': 'lonely',
    'neutral': 'happy'
}


def detect_emotion(text):
    # Translate if needed
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, dest='en').text
    except:
        pass

    # Tokenize & predict
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    ranking = scores.argsort()[::-1]
    top_emotion = model.config.id2label[ranking[0]].lower()
    return emotion_alias.get(top_emotion, top_emotion)

def fetch_poster(title):
    api_key = "ca21149d"  # Replace with real OMDb key
    try:
        response = requests.get(f"http://www.omdbapi.com/?t={title}&apikey={api_key}")
        data = response.json()
        return data.get("Poster", "N/A")
    except:
        return "N/A"

def recommend_movies(emotion, language=None, top_n=5):
    matches = df[df['emotion'].str.lower() == emotion.lower()]

    if language:
        matches = matches[matches['language'].str.lower() == language.lower()]

    if emotion in genre_filter:
        matches = matches[matches['genre'].isin(genre_filter[emotion])]

    # Fallback if no matches after filtering
    if matches.empty and emotion in genre_filter:
        matches = df[df['emotion'].str.lower() == emotion.lower()]
        if language:
            matches = matches[matches['language'].str.lower() == language.lower()]

    top_rated = matches.sort_values(by='rating', ascending=False)
    return top_rated[['title', 'genre', 'language', 'rating', 'review']].head(top_n)

def main():
    print("\n🎬 Welcome to Mood2Movie!")
    feeling = input("How are you feeling today?\n> ")
    emotion = detect_emotion(feeling)
    print(f"\nDetected emotion: **{emotion.upper()}**")

    languages = sorted(df['language'].dropna().unique())
    print(f"\nAvailable languages: {', '.join(languages)}")
    user_lang = input("Choose your preferred movie language [or press Enter to see all]:\n> ")
    if user_lang.strip() == "":
        user_lang = None

    recommendations = recommend_movies(emotion, user_lang)

    if recommendations.empty:
        print("😞 Sorry, we couldn't find any movies matching that mood and language.")
    else:
        print("\n🎥 Recommended Movies:\n")
        print(f"Found {len(recommendations)} matching movies.\n")
        for _, row in recommendations.iterrows():
            poster = fetch_poster(row['title'])
            print(f"- {row['title']} ({row['genre']} - {row['language']}) - {row['rating']}\n  \"{row['review']}\"\n  Poster: {poster}\n")

if __name__ == '__main__':
    main()