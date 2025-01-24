import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_researcher_features(researchers_df):
    """Extract features from researchers data."""
    # TF-IDF for research concepts
    tfidf = TfidfVectorizer(stop_words="english", max_features=50)
    researcher_concepts = (researchers_df["concept_1"].fillna("") + " " + researchers_df["concept_2"].fillna(""))
    researchers_df["concept_embeddings"] = tfidf.fit_transform(researcher_concepts).toarray().tolist()

    # Normalize numerical features
    researchers_df["total_citations"] = researchers_df["total_citations"].fillna(0)
    researchers_df["h_index"] = researchers_df["h_index"].fillna(0)
    return researchers_df