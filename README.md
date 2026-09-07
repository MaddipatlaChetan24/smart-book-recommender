<div align="center">

# Smart Book Recommender

Personalized Machine Learning Book Recommendation System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge\&logo=flask\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge\&logo=docker\&logoColor=white)
![License](https://img.shields.io/badge/MIT-License-blue?style=for-the-badge)

</div>

# Overview

Smart Book Recommender is a machine learning-powered recommendation system designed to provide personalized book suggestions based on user preferences, reading patterns, and book similarity.

The system combines **content-based recommendation techniques** with pre-trained machine learning models to efficiently discover books that match a user's interests.

The application allows users to:

* Discover personalized book recommendations
* Find books similar to a selected title
* Explore popular books
* Search and select books through an intuitive interface
* Generate recommendations using pre-trained models
* Receive recommendations without real-time model retraining

---

# Features

## Personalized Recommendations

The system analyzes book information and user preferences to generate relevant recommendations tailored to individual interests.

## Similar Book Discovery

Users can select a book and discover other books with similar characteristics using text-based feature extraction and similarity analysis.

## Popular Books

The application provides a collection of popular books based on available user engagement and rating information.

## Machine Learning Recommendations

The recommendation pipeline uses **TF-IDF vectorization** and **cosine similarity** to identify books with similar content characteristics.

## Fast Inference

Pre-trained models and processed datasets are used to generate recommendations efficiently without retraining the machine learning pipeline for every request.

## Web Interface

A Flask-based web interface provides a simple way for users to search, select, and explore recommended books.

---

# Recommendation Architecture

```text
                    User
                     │
                     ▼
              Flask Web Interface
                     │
                     ▼
              Search / Select Book
                     │
                     ▼
              Book Feature Extraction
                     │
                     ▼
              TF-IDF Vectorization
                     │
                     ▼
             Cosine Similarity
                     │
                     ▼
              Similarity Ranking
                     │
                     ▼
             Top Book Recommendations
```

---

# Recommendation Approach

## Content-Based Filtering

The system uses information associated with books to determine their similarity.

The recommendation pipeline includes:

1. Book metadata preprocessing
2. Text feature extraction
3. TF-IDF vectorization
4. Cosine similarity calculation
5. Similarity-based ranking
6. Selection of top recommendations

This allows the system to recommend books that share similar characteristics with a book selected by the user.

## Popularity-Based Recommendations

The system also uses popularity information from the available book and rating data to identify books with strong user engagement.

This provides users with an additional way to discover highly rated or frequently interacted-with books.

---

# Project Structure

```text
Smart-Book-Recommender/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── LICENSE
├── README.md
│
├── Datasets/
│   ├── README.md
│   └── book and rating datasets
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── popular_books.html
│   ├── collaborative.html
│   └── personal.html
│
└── *.pkl
    ├── popularity_model.pkl
    └── collab_model.pkl
```

---

# Technology Stack

**Programming Language**

* Python

**Backend**

* Flask

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy

**Recommendation Techniques**

* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Filtering
* Popularity-Based Recommendation

**Model Serialization**

* Joblib / Pickle

**Frontend**

* HTML5
* CSS3
* JavaScript

**Deployment**

* Docker
* Gunicorn
* Render

---

# Installation

## Clone Repository

```bash
git clone https://github.com/MaddipatlaChetan24/smart-book-recommender.git

cd smart-book-recommender
```

## Create Virtual Environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Start the Flask application.

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

Open the URL in a browser to access the recommendation system.

---

# Docker Deployment

Build the Docker image:

```bash
docker build -t smart-book-recommender .
```

Run the container:

```bash
docker run -p 5000:5000 smart-book-recommender
```

The application will then be available at:

```text
http://127.0.0.1:5000
```

---

# Dataset

The system uses book-related datasets containing information such as:

* Book titles
* Authors
* Publishers
* Publication years
* Genres and categories
* User ratings
* User interactions

The data is preprocessed before being used by the recommendation pipeline.

Textual book information is transformed into numerical representations using **TF-IDF**, enabling efficient similarity calculations.

---

# Machine Learning Pipeline

```text
Raw Book Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Text Feature Extraction
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Similarity Matrix
       │
       ▼
Pre-trained Model
       │
       ▼
Recommendation Engine
       │
       ▼
Top Recommended Books
```

---

# Roadmap

* Hybrid collaborative + content-based recommendations
* User authentication and persistent profiles
* Personalized recommendation history
* Advanced NLP for book descriptions and reviews
* Sentiment analysis on user reviews
* Goodreads / OpenLibrary API integration
* Improved mobile responsiveness
* Recommendation explanation system
* REST API for external applications
* Advanced recommendation ranking
* Docker-based production deployment improvements

---

# Contributing

Contributions are welcome.

```bash
git checkout -b feature/new-feature
git commit -m "Add new feature"
git push origin feature/new-feature
```

Then open a Pull Request.

---

# License

This project is distributed under the MIT License.
