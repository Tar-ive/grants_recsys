import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_grant_features(grants_df):
    """Extract features from grants data."""
    # TF-IDF for grant descriptions
    tfidf = TfidfVectorizer(stop_words="english", max_features=100)
    grant_descriptions = grants_df["summary_description"].fillna("")
    grants_df["description_embeddings"] = tfidf.fit_transform(grant_descriptions).toarray().tolist()

    # Normalize numerical features
    grants_df["award_ceiling"] = grants_df["award_ceiling"].fillna(0)
    grants_df["award_floor"] = grants_df["award_floor"].fillna(0)
    return grants_df