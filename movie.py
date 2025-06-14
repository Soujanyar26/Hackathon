import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from langdetect import detect
from googletrans import Translator
from textwrap import fill
from datetime import datetime
import random

# Load dataset
df = pd.read_csv(r"E:\hackathon\movie_dataset.csv")

# Load emotion model
MODEL = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Translator for multilingual support
translator = Translator()

# Map detected emotions to genres
genre_filter = {
    'happy': ['Comedy', 'Romance', 'Musical'],
    'sad': ['Drama', 'Romance', 'Biography'],
    'lonely': ['Drama', 'Romance', 'Adventure'],
    'fear': ['Comedy', 'Sci-Fi', 'Mystery'],
    'surprise': ['Mystery', 'Thriller', 'Horror']
}

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

emotion_emoji = {
    'happy': '😊',
    'sad': '😢',
    'lonely': '😔',
    'fear': '😨',
    'surprise': '😲'
}

def detect_emotion(text):
    try:
        lang = detect(text)
        if lang != 'en':
            text = translator.translate(text, dest='en').text
    except:
        pass
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    ranking = scores.argsort()[::-1]
    top_emotion = model.config.id2label[ranking[0]].lower()
    return emotion_alias.get(top_emotion, top_emotion)

def fetch_poster(title):
    api_key = "ca21149d"
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
    if matches.empty and emotion in genre_filter:
        matches = df[df['emotion'].str.lower() == emotion.lower()]
        if language:
            matches = matches[matches['language'].str.lower() == language.lower()]
    top_rated = matches.sort_values(by='rating', ascending=False)
    return top_rated[['title', 'genre', 'language', 'rating', 'review']].head(top_n)

def get_surprise_movies(language=None, top_n=5):
    genres = df['genre'].dropna().unique()
    surprise_genre = random.choice(genres)
    matches = df[df['genre'] == surprise_genre]
    if language:
        matches = matches[matches['language'].str.lower() == language.lower()]
    return matches.sort_values(by='rating', ascending=False).head(top_n), surprise_genre

def print_boxed_movie(index, row, poster_url):
    title = f"{index+1}. 🎬 {row['title']} ({row['genre']} - {row['language']})"
    rating = f"⭐ Rating: {row['rating']}"
    review = f"💬 Review: {fill(row['review'], width=70)}"
    poster = f"🖼️ Poster: {poster_url}"
    line = "─" * 70
    print(f"\n{line}\n{title}\n{rating}\n{review}\n{poster}\n{line}")

def main():
    print("\n🎞️ Welcome to Mood2Movie!")
    feeling = input("🧠 How are you feeling today?\n> ")
    emotion = detect_emotion(feeling)
    emoji = emotion_emoji.get(emotion, "")
    print(f"\n🧠 Detected Emotion: **{emotion.upper()}** {emoji}")

    # Log the mood
    with open("mood_history.txt", "a") as log:
        log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M')},{emotion},{feeling}\n")

    languages = sorted(df['language'].dropna().unique())
    print(f"\n🌐 Available Languages: {', '.join(languages)}")
    user_lang = input("🌍 Choose your preferred language [press Enter for all]:\n> ")
    user_lang = user_lang if user_lang.strip() != "" else None

    # Surprise option
    surprise = input("\n🎁 Do you want a surprise genre recommendation? (y/n):\n> ").lower()
    if surprise == 'y':
        surprise_recos, genre = get_surprise_movies(user_lang)
        print(f"\n🎉 Surprise Genre: {genre.upper()} — Here are your movies!\n")
        for i, row in surprise_recos.iterrows():
            poster = fetch_poster(row['title'])
            print_boxed_movie(i, row, poster)
        return

    # Normal recommendations
    recommendations = recommend_movies(emotion, user_lang)
    if recommendations.empty:
        print("\n😞 Sorry, no matching movies found for your emotion and language.")
    else:
        print(f"\n🎯 Found {len(recommendations)} movie(s) for you:\n")
        for i, row in recommendations.iterrows():
            poster = fetch_poster(row['title'])
            print_boxed_movie(i, row, poster)

    # Feedback
    feedback = input("\n📝 Did these recommendations match your mood? (yes/no):\n> ")
    with open("feedback_log.txt", "a") as f:
        f.write(f"{emotion},{user_lang},{feedback}\n")

if __name__ == '__main__':
    main()
