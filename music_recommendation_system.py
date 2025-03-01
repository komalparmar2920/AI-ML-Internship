import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("C:/Users/Lenovo/Downloads/spotify_millsongdata.csv")
df = df.dropna().reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])


def get_recommendations(song_title, df, tfidf_matrix):
    idx = df[df["song"].str.lower() == song_title.lower()].index
    if len(idx) == 0:
        return "Song not found in database. Try another song."

    idx = idx[0]
    song_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(song_vec, tfidf_matrix).flatten()

    song_indices = sim_scores.argsort()[-6:-1][::-1]

    return df.iloc[song_indices][["artist", "song"]]


song = input("Enter song name: ").strip()
recommendations = get_recommendations(song, df, tfidf_matrix)
print(recommendations)
