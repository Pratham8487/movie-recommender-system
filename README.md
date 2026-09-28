🎬 Movie Recommendation System

A content-based movie recommendation system built with Python,
Pandas, NLTK, and Scikit-learn.

The system recommends 5 movies similar to a movie selected by the
user. Instead of using user ratings or collaborative filtering, it
analyzes the textual/content information associated with each movie,
converts that information into numerical vectors, calculates the
similarity between movies, and returns the most similar titles.

📌 Project Overview

The recommendation pipeline used in this project is:

TMDB Movie Dataset
        │
        ├── tmdb_5000_movies.csv
        │
        └── tmdb_5000_credits.csv
                │
                ▼
        Merge both datasets
                │
                ▼
        Select useful movie features
                │
                ▼
        Clean & transform JSON-like columns
                │
                ▼
        Extract:
        • Overview
        • Genres
        • Keywords
        • Top 3 Cast Members
        • Director
                │
                ▼
        Combine everything into "tags"
                │
                ▼
        Stemming + Lowercasing
                │
                ▼
        CountVectorizer
        max_features = 5000
                │
                ▼
        Movie feature vectors
                │
                ▼
        Cosine Similarity
                │
                ▼
        Similarity Matrix
                │
                ▼
        Select 5 most similar movies
                │
                ▼
        Recommendations

✨ Features

Content-based movie recommendation

Uses movie metadata rather than user-rating history

Combines multiple movie attributes into one text representation

Extracts the top 3 cast members

Extracts the movie director

Uses NLTK Porter Stemming

Uses Scikit-learn CountVectorizer

Uses English stop-word removal

Calculates movie-to-movie similarity using cosine similarity

Returns the top 5 similar movies

Saves the processed movie data and similarity matrix using Pickle

📂 Dataset

This project uses the TMDB 5000 Movie Dataset, consisting of two CSV
files:

1. tmdb_5000_movies.csv

This file contains movie-level metadata such as:

Movie ID

Title

Overview

Genres

Keywords

Budget

Popularity

Release date

Revenue

Runtime

Vote average

Vote count

Production companies

Production countries

Spoken languages

And other movie metadata

2. tmdb_5000_credits.csv

This file contains credit information such as:

Movie title

Cast

Crew

Character information

Crew roles

Dataset Source

The dataset used in this project is available on Kaggle:

TMDB 5000 Movie Dataset:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download

Download the dataset and place both CSV files in the project directory.

Expected files:

tmdb_5000_movies.csv
tmdb_5000_credits.csv

🧠 How the Recommendation System Works

This is a content-based filtering recommendation system.

The basic idea is:

If two movies have similar content/features, they should receive a
high similarity score and can be recommended to each other.

For example, if a movie contains tags related to:

action
adventure
superhero
marvel
ironman
robert downey jr

then movies containing similar tags will have a higher similarity score.

The system does not learn individual user preferences. It only
compares the content representation of movies.

🔎 Step-by-Step Implementation

1. Import Required Libraries

The notebook starts by importing the libraries required for data
processing and text manipulation.

import pandas as pd
import numpy as np
import ast
import nltk

Libraries

Library                             Purpose

Pandas                              Data loading and manipulation

NumPy                               Numerical operations

AST                                 Convert string representations of
Python lists/dictionaries into
Python objects

Later, Scikit-learn is used for vectorization and similarity
calculation.

2. Load the Dataset

The two CSV files are loaded using Pandas:

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

At this point there are two separate DataFrames:

movies
credits

The movie dataset contains movie metadata, while the credits dataset
contains cast and crew information.

3. Merge the Two Datasets

The datasets are merged using the movie title:

movies = movies.merge(credits, on='title')

This combines movie information and credit information into one
DataFrame.

Conceptually:

Movie Metadata
      +
Cast / Crew Metadata
      ↓
Combined Movie Dataset

The project then has access to both movie content and people involved in
the movie.

4. Select Relevant Features

The notebook keeps only the columns required for the recommendation
system:

movies = movies[
    ['movie_id', 'title', 'overview', 'genres',
     'keywords', 'cast', 'crew']
]

The selected features are:

movie_id

Unique identifier for the movie.

title

Movie title used to identify the movie during recommendation.

overview

Textual description of the movie.

genres

Movie categories such as:

Action
Adventure
Comedy
Drama
Science Fiction

keywords

Keywords describing important concepts, themes, or topics associated
with the movie.

cast

Actors appearing in the movie.

crew

Used to extract the director.

These fields are selected because they contain information that can
describe the movie's content.

🧹 5. Check Missing Values

The notebook checks missing values:

movies.isnull().sum()

Then rows containing missing values are removed:

movies.dropna(inplace=True)

This ensures that the fields required for creating the recommendation
tags are available.

🔍 6. Check Duplicate Movies

The notebook also checks for duplicated rows:

movies.duplicated().sum()

This is a basic data-quality check before continuing with feature
engineering.

🧩 7. Convert JSON-Like Data into Lists

Several columns in the dataset are stored as strings representing lists
of dictionaries.

For example, genres can look conceptually like:

[
    {'id': 28, 'name': 'Action'},
    {'id': 12, 'name': 'Adventure'}
]

The notebook uses ast.literal_eval() to convert the string into a
Python object.

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

The function extracts only the name values.

For example:

[
    {"id": 28, "name": "Action"},
    {"id": 12, "name": "Adventure"}
]

becomes:

["Action", "Adventure"]

The function is applied to:

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

👥 8. Extract the Top 3 Cast Members

The recommendation system does not use the complete cast list.

Instead, it extracts the first three cast members:

def convert_cast(obj):
    l = []
    counter = 0

    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break

    return l

Then:

movies['cast'] = movies['cast'].apply(convert_cast)

For example, if a movie has:

Actor A
Actor B
Actor C
Actor D
Actor E

the recommendation system keeps:

Actor A
Actor B
Actor C

This reduces the amount of information used in the movie representation
while retaining the leading cast members provided by the dataset.

🎬 9. Extract the Director

The crew field contains many different crew roles.

The project specifically searches for the person whose job is:

Director

The function is:

def fetch_director(obj):
    l = []

    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            l.append(i['name'])
            break

    return l

Then:

movies['crew'] = movies['crew'].apply(fetch_director)

The crew column is therefore transformed into a list containing the
director's name.

📝 10. Convert Movie Overview into Words

The movie overview is converted from a sentence into a list of words:

movies['overview'] = movies['overview'].apply(lambda x: x.split())

For example:

"A superhero saves the world"

becomes approximately:

["A", "superhero", "saves", "the", "world"]

This makes it possible to combine the overview with the other tag-based
features.

🧼 11. Remove Spaces from Names and Tags

The notebook removes spaces from values in:

Genres

Keywords

Cast

Crew

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(' ','') for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(' ','') for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(' ','') for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(' ','') for i in x]
)

This is important for multi-word entities.

For example:

Robert Downey Jr

becomes:

RobertDowneyJr

and:

Science Fiction

becomes:

ScienceFiction

This helps the vectorizer treat these names/phrases as individual tags.

🏷️ 12. Create the tags Column

This is one of the most important steps in the project.

The project combines:

Overview

Genres

Keywords

Cast

Director

into one column:

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

The result is a single list containing the movie's descriptive
information.

Conceptually:

Movie
│
├── Overview
├── Genres
├── Keywords
├── Top 3 Cast
└── Director
        │
        ▼
      TAGS

For example:

overview words
+
Action
Adventure
+
superhero
marvel
+
Actor1
Actor2
Actor3
+
Director

becomes one combined representation.

📦 13. Create a Smaller Movie DataFrame

The project creates:

new_movies = movies[['movie_id', 'title', 'tags']]

The final recommendation dataset therefore contains:

movie_id
title
tags

This is the main DataFrame used by the recommendation engine.

🌱 14. Apply Stemming

The project uses NLTK's Porter Stemmer:

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

A custom function is created:

def stem(text):
    y = []

    if isinstance(text, list):
        for i in text:
            y.append(ps.stem(i))
    else:
        for i in text.split():
            y.append(ps.stem(i))

    return " ".join(y)

Then it is applied:

new_movies['tags'] = new_movies['tags'].apply(stem)

What is stemming?

Stemming reduces words to a common root/stem.

For example, words with related forms may be reduced to similar stems.

This helps the text representation treat related word forms more
similarly.

🔡 15. Convert Tags to Lowercase

The project then converts the tags to lowercase:

new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())

For example:

Action Adventure Marvel

becomes:

action adventure marvel

This avoids treating uppercase and lowercase versions as separate terms.

🔢 16. Text Vectorization

The project uses Scikit-learn's CountVectorizer:

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

This is the step where text is converted into numerical vectors.

Computers cannot directly calculate cosine similarity between raw text,
so each movie's tags need to be represented numerically.

What Does CountVectorizer Do?

Suppose we have:

Movie A:
action hero marvel

Movie B:
action hero dc

Movie C:
romance drama love

The vectorizer builds a vocabulary from the text.

It then represents each movie using the occurrence/count of vocabulary
terms.

Conceptually:

             action hero marvel dc romance drama love
Movie A        1     1    1    0    0      0    0
Movie B        1     1    0    1    0      0    0
Movie C        0     0    0    0    1      1    1

The actual vectors in this project can contain up to 5,000 features
because:

max_features=5000

is specified.

English stop words are removed using:

stop_words='english'

🧮 17. Generate Movie Vectors

The tags are converted into numerical vectors:

vectors = cv.fit_transform(new_movies['tags']).toarray()

After this operation:

Movie Tags
    ↓
CountVectorizer
    ↓
Numerical Vectors

Each row corresponds to a movie.

Each column corresponds to a vocabulary feature.

The resulting matrix can therefore be thought of as:

              Feature 1  Feature 2  Feature 3 ... Feature N
Movie 1           ...        ...        ...          ...
Movie 2           ...        ...        ...          ...
Movie 3           ...        ...        ...          ...
...

📐 18. Calculate Cosine Similarity

The project uses:

from sklearn.metrics.pairwise import cosine_similarity

Then calculates:

similarity = cosine_similarity(vectors)

This creates a movie-to-movie similarity matrix.

What Is Cosine Similarity?

Cosine similarity measures how similar two vectors are based on the
angle between them.

The formula is:

             A · B
cos(θ) = --------------
          ||A|| ||B||

Where:

A = vector representing movie A

B = vector representing movie B

A · B = dot product

||A|| and ||B|| = vector magnitudes

A higher cosine similarity means the movie vectors point in more similar
directions.

Conceptually:

Movie A  ──────────────►
             small angle
                  ╲
                   ╲
                    ─────────► Movie B

             High similarity

Whereas unrelated movies will generally have a larger angle between
their vectors.

🧩 19. Similarity Matrix

After calculating cosine similarity:

similarity = cosine_similarity(vectors)

the project has a matrix where:

similarity[i][j]

represents the similarity between movie i and movie j.

Conceptually:

             Movie 1  Movie 2  Movie 3  Movie 4
Movie 1        1.00     0.72     0.15     0.34
Movie 2        0.72     1.00     0.21     0.41
Movie 3        0.15     0.21     1.00     0.08
Movie 4        0.34     0.41     0.08     1.00

The diagonal values represent a movie compared with itself and are
expected to be the highest similarity.

🎯 20. Recommendation Logic

The recommendation function is:

def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    for i in movies_list:
        print(new_movies.iloc[i[0]].title)

Let's break it down.

Step 20.1 --- Find the Movie Index

movie_index = new_movies[
    new_movies['title'] == movie
].index[0]

If the user enters:

Pirates of the Caribbean: At World's End

the system searches for that title and gets its DataFrame index.

Step 20.2 --- Get Its Similarity Scores

distances = similarity[movie_index]

This retrieves the similarity score of the selected movie against every
movie in the dataset.

Conceptually:

Selected Movie
      │
      ▼
Similarity scores
      │
      ├── Movie A → 0.12
      ├── Movie B → 0.85
      ├── Movie C → 0.43
      ├── Movie D → 0.91
      └── ...

Step 20.3 --- Sort Movies by Similarity

The movies are sorted in descending order:

sorted(
    list(enumerate(distances)),
    reverse=True,
    key=lambda x: x[1]
)

The highest similarity scores appear first.

Step 20.4 --- Remove the Selected Movie

The first item is skipped:

[1:6]

Why?

Because a movie is maximally similar to itself.

Without removing the first item, the selected movie itself would appear
as the first recommendation.

Step 20.5 --- Return the Top 5

The slice:

[1:6]

selects exactly five movies after excluding the selected movie.

Therefore:

Selected Movie
      ↓
Similarity Scores
      ↓
Sort Descending
      ↓
Remove Selected Movie
      ↓
Take Top 5
      ↓
Print Recommendations

▶️ 21. Example Usage

The notebook calls:

recommend("Pirates of the Caribbean: At World's End")

The function then finds the five movies with the highest cosine
similarity to the selected movie based on the generated tags.

The recommendations are based on the movie's content representation, not
on individual user ratings.

💾 22. Save the Recommendation Data

The notebook uses Python's pickle module:

import pickle

The processed movie data is saved using:

pickle.dump(
    new_movies.to_dict(),
    open('movie_dict.pkl', 'wb')
)

This creates:

movie_dict.pkl

It stores the processed movie information needed by the recommendation
application.

💾 23. Save the Similarity Matrix

The similarity matrix is also saved:

pickle.dump(
    similarity,
    open('similarity.pkl', 'wb')
)

This creates:

similarity.pkl

It contains the precomputed movie-to-movie similarity matrix.

Saving the matrix means the similarity calculation does not have to be
repeated every time recommendations are requested.

🏗️ Project Architecture

The project can be understood as three major stages.

Stage 1 --- Data Preparation

TMDB CSV Files
      ↓
Load with Pandas
      ↓
Merge Movies + Credits
      ↓
Select Features
      ↓
Clean Missing Data
      ↓
Parse Genres / Keywords / Cast / Crew

Stage 2 --- Feature Engineering

Overview
Genres
Keywords
Top 3 Cast
Director
      ↓
Combine
      ↓
Tags
      ↓
Stemming
      ↓
Lowercase

Stage 3 --- Recommendation Engine

Tags
  ↓
CountVectorizer
  ↓
Movie Vectors
  ↓
Cosine Similarity
  ↓
Similarity Matrix
  ↓
Sort Similarity Scores
  ↓
Top 5 Movies

🧪 Why This Is a Content-Based Recommendation System

This project is classified as a content-based recommendation system
because recommendations are generated from the characteristics of the
movies themselves.

The system does not require:

User accounts

User ratings

Watch history

User-to-user similarity

Collaborative filtering

Instead, it asks:

"Which movies have content/features most similar to this movie?"

🔬 Feature Engineering Summary

The final movie representation is created from:

Movie Overview
      +
Genres
      +
Keywords
      +
Top 3 Cast Members
      +
Director
      ↓
    Tags
      ↓
Stemming
      ↓
Lowercase
      ↓
CountVectorizer
      ↓
Numerical Vector

This feature engineering step is the foundation of the recommendation
quality.

📦 Technologies Used

Technology         Usage

Python             Core programming language
Pandas             Data loading and manipulation
NumPy              Numerical operations
NLTK               Porter stemming
Scikit-learn       CountVectorizer and cosine similarity
AST                Parsing JSON-like string data
Pickle             Saving processed data and similarity matrix
Jupyter Notebook   Development and experimentation
Kaggle             Dataset source

📁 Suggested Project Structure

movie-recommendation-system/
│
├── movie_recommendation.ipynb
│
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
│
├── movie_dict.pkl
├── similarity.pkl
│
└── README.md

The notebook performs the data preprocessing and model preparation,
while the Pickle files store the processed data required for a
recommendation application.

⚙️ Installation

1. Clone the Repository

git clone https://github.com/Pratham8487/movie-recommender-system.git
cd movie-recommender-system

2. Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python3 -m venv venv
source venv/bin/activate

📥 Install Dependencies

Install the required Python libraries:

pip install pandas numpy nltk scikit-learn jupyter

If you are using the notebook, launch Jupyter:

jupyter notebook

Then open:

movie_recommendation.ipynb

▶️ Running the Project

Follow these steps:

Step 1

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download

Step 2

Place:

tmdb_5000_movies.csv
tmdb_5000_credits.csv

in the project directory.

Step 3

Open:

movie_recommendation.ipynb

Step 4

Run the notebook cells in order.

Step 5

The notebook will:

Load data
   ↓
Merge datasets
   ↓
Clean data
   ↓
Create tags
   ↓
Stem text
   ↓
Vectorize text
   ↓
Calculate cosine similarity
   ↓
Create recommendation engine
   ↓
Save pickle files

Step 6

Use:

recommend("Movie Name")

to generate five recommendations.

🧮 Recommendation Algorithm in Simple Terms

Suppose the user selects:

Movie A

The system does this:

1. Find Movie A
        ↓
2. Get Movie A's tags
        ↓
3. Convert all movie tags into vectors
        ↓
4. Compare Movie A's vector with every movie vector
        ↓
5. Calculate cosine similarity
        ↓
6. Sort movies by similarity score
        ↓
7. Remove Movie A itself
        ↓
8. Select the top 5
        ↓
9. Return movie titles

So the core algorithm can be summarized as:

Content → Tags → Vector → Similarity → Ranking → Recommendation

📊 What Makes Two Movies Similar?

Two movies can receive a high similarity score when they share similar
information in their generated tags.

For example, movies may share:

Similar genres

Similar keywords

Similar actors

Same director

Similar words in their overview

Because all of these are combined into the tags field, they influence
the final vector representation and therefore the cosine similarity.

⚠️ Important Characteristics and Limitations

This implementation is a learning-focused content-based recommender and
has some limitations.

1. No User Personalization

The system recommends movies similar to the selected movie.

It does not know whether a particular user personally likes or dislikes
those movies.

2. No Collaborative Filtering

There is no user-item rating matrix and no collaborative filtering.

3. Exact Title Lookup

The recommendation function searches for:

new_movies['title'] == movie

Therefore, the input needs to match a movie title present in the
dataset.

4. Duplicate Titles

Because the lookup uses the first matching index:

.index[0]

movies with duplicate titles may require additional handling in a
production application.

5. Count-Based Representation

The implementation uses CountVectorizer, so the representation is
based on vocabulary term counts rather than TF-IDF or semantic
embeddings.

6. Fixed Vocabulary Size

Only the top:

5,000

features are retained by CountVectorizer.

7. Similarity Matrix Size

The notebook calculates pairwise similarity for the complete movie
collection.

For a much larger dataset, storing a full movie-by-movie similarity
matrix can become memory-intensive.

🚀 Possible Future Improvements

The current implementation can be extended in several ways.

1. TF-IDF Vectorization

Replace CountVectorizer with TfidfVectorizer to give different
importance to terms based on their frequency across the dataset.

2. Semantic Embeddings

Use modern text embeddings to capture semantic meaning rather than only
matching vocabulary terms.

Possible approaches include:

Sentence Transformers

Transformer embeddings

Other embedding models

3. Better User Personalization

Combine content-based recommendations with:

User ratings

Watch history

Likes/dislikes

Favorite genres

User profiles

This can lead to a hybrid recommendation system.

4. Improved Search

Instead of requiring an exact title, add:

Fuzzy title matching

Search suggestions

Autocomplete

Movie IDs

5. Production API

The recommendation logic could be exposed through an API using
frameworks such as:

FastAPI

or:

Flask

6. Web Interface

A frontend can be added using:

HTML/CSS/JavaScript

or a framework such as:

React

The user could search for a movie and receive five recommendations
through a web interface.

🔄 End-to-End Example

Suppose the user enters:

recommend("Pirates of the Caribbean: At World's End")

The system performs:

Pirates of the Caribbean: At World's End
                    │
                    ▼
              Find movie index
                    │
                    ▼
             Retrieve similarity row
                    │
                    ▼
          Compare against all movies
                    │
                    ▼
             Sort similarity scores
                    │
                    ▼
             Remove selected movie
                    │
                    ▼
             Select top 5 movies
                    │
                    ▼
              Print movie titles

The recommendations are therefore the five movies with the highest
calculated content similarity to the selected movie.

📌 Key Concepts Demonstrated

This project demonstrates several important Machine Learning and Data
Science concepts:

Data loading

Data merging

Data cleaning

Missing-value handling

Duplicate checking

Feature selection

Feature engineering

JSON-like data parsing

Natural Language Processing

Stemming

Text normalization

Bag-of-Words representation

Count Vectorization

Vector similarity

Cosine similarity

Ranking

Content-based recommendation

Model/data serialization with Pickle

📚 Learning Takeaway

The most important concept in this project is that recommendation
quality depends heavily on how the items are represented.

Here, each movie is represented using a combination of:

Overview
+ Genres
+ Keywords
+ Top 3 Cast
+ Director

That combined representation is converted into a numerical vector.

The system then compares those vectors:

Movie Content
     ↓
Text Representation
     ↓
Numerical Vector
     ↓
Cosine Similarity
     ↓
Similarity Ranking
     ↓
Top 5 Recommendations

This makes the project a practical example of how Natural Language
Processing and vector similarity can be used to build a recommendation
engine.

👨‍💻 Author

Pratham Shah

GitHub: https://github.com/Pratham8487

Portfolio: https://prathamshah.me

📄 Dataset Attribution

Dataset: TMDB 5000 Movie Dataset

Source: Kaggle

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download

The dataset is used as the source of movie metadata and credits for this
project.

⭐ Project Summary

In one sentence:

This project builds a content-based movie recommendation system by
combining movie metadata into text tags, converting those tags into
numerical vectors using CountVectorizer, measuring movie similarity
using cosine similarity, and recommending the five most similar
movies.
