import logging
import os
from collections import Counter

import joblib
import numpy as np
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load models
top_100_books = joblib.load('popularity_model.pkl')
collab_data = joblib.load('collab_model.pkl')
tfidf = collab_data['tfidf_vectorizer']
tfidf_matrix = collab_data['tfidf_matrix']
reduced_df = collab_data['filtered_df']

# Rating is used as a light tie-breaker on top of content similarity so that,
# among near-equally-similar books, better-reviewed ones surface first.
_RATING_BOOST = (reduced_df['rating'].fillna(reduced_df['rating'].mean()) / 5).to_numpy()


def _top_genres(n=20):
    """Most frequent individual genres, used to power genre-based onboarding
    for users who have no reading history yet (the "cold start" case)."""
    counter = Counter()
    for genre_str in reduced_df['genre'].dropna():
        for part in genre_str.split(','):
            part = part.strip()
            if part:
                counter[part] += 1
    return [genre for genre, _ in counter.most_common(n)]


TOP_GENRES = _top_genres()


def recommend_from_titles(titles, top_n=10):
    """Content-based recommendations aggregated across one or more seed
    titles. Returns [] if none of the titles are found in the catalog."""
    all_sim_scores = np.zeros(len(reduced_df))
    matched_any = False

    for title in titles:
        matches = reduced_df.index[reduced_df['title'] == title]
        if len(matches) == 0:
            continue
        matched_any = True
        idx = matches[0]
        all_sim_scores += cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    if not matched_any:
        return []

    hybrid_scores = all_sim_scores * (0.85 + 0.15 * _RATING_BOOST)
    selected_set = set(titles)

    results = []
    for i in hybrid_scores.argsort()[::-1]:
        if hybrid_scores[i] <= 0:
            break
        title = reduced_df.iloc[i]['title']
        if title in selected_set:
            continue
        results.append(reduced_df.iloc[i])
        if len(results) >= top_n:
            break
    return results


def popularity_fallback(n=12, genres=None):
    """Books to show someone with no (usable) reading history: highest
    rated/most reviewed titles, optionally narrowed to chosen genres.
    This is the primary defense against the cold-start problem."""
    df = reduced_df
    if genres:
        chosen = set(genres)
        mask = df['genre'].apply(lambda g: bool({p.strip() for p in g.split(',')} & chosen))
        subset = df[mask]
        if subset.empty:
            subset = df
    else:
        subset = df

    subset = subset.sort_values(by=['rating', 'reviews'], ascending=False)
    return [subset.iloc[i] for i in range(min(n, len(subset)))]


def find_title_matches(query, titles, limit=10):
    """Rank substring matches, preferring titles that start with the query
    over titles that merely contain it."""
    query = query.lower()
    starts = [t for t in titles if t.lower().startswith(query)]
    contains = [t for t in titles if query in t.lower() and t not in starts]
    return (starts + contains)[:limit]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/popular_books')
def popular_books():
    return render_template('popular_books.html', books=top_100_books)


@app.route('/collaborative', methods=['GET', 'POST'])
def collaborative():
    book_titles = reduced_df['title'].dropna().unique()
    similar_books = []
    selected_title = None
    search_query = None
    fallback_books = []

    if request.method == 'POST':
        if request.form.get('book_title'):
            selected_title = request.form.get('book_title')
        elif request.form.get('search_query'):
            search_query = request.form.get('search_query')
            matches = find_title_matches(search_query, book_titles, limit=1)
            if matches:
                selected_title = matches[0]
            else:
                fallback_books = popularity_fallback(n=6)

        if selected_title:
            similar_books = recommend_from_titles([selected_title], top_n=5)
            if not similar_books:
                fallback_books = popularity_fallback(n=6)

    return render_template('collaborative.html',
                            book_titles=book_titles,
                            similar_books=similar_books,
                            selected_title=selected_title,
                            search_query=search_query,
                            fallback_books=fallback_books)


@app.route('/personal', methods=['GET', 'POST'])
def personal_recommendations():
    book_titles = reduced_df['title'].dropna().unique()
    recommended_books = []
    selected_titles = []
    selected_genres = []
    is_cold_start = False
    is_fallback = False

    if request.method == 'POST':
        selected_titles = request.form.getlist('book_titles')
        selected_genres = request.form.getlist('genres')

        if selected_titles:
            recommended_books = recommend_from_titles(selected_titles, top_n=12)
            if not recommended_books:
                # Titles were submitted but none matched the catalog -
                # don't leave the user with a blank page.
                recommended_books = popularity_fallback(n=12, genres=selected_genres or None)
                is_fallback = True
        elif selected_genres:
            # No reading history given at all: classic cold-start user.
            # Use their stated genre preferences instead.
            recommended_books = popularity_fallback(n=12, genres=selected_genres)
            is_cold_start = True

    return render_template('personal.html',
                            book_titles=book_titles,
                            top_genres=TOP_GENRES,
                            recommended_books=recommended_books,
                            selected_titles=selected_titles,
                            selected_genres=selected_genres,
                            is_cold_start=is_cold_start,
                            is_fallback=is_fallback)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)
