from lightgbm import LGBMRanker
import numpy as np
from features.grants import extract_grant_features
from features.researchers import extract_researcher_features
from raw_data_sources.grants import fetch_grants
from raw_data_sources.researchers import fetch_researchers

def train_lightgbm_ranking_model():
    grants_df = fetch_grants()
    researchers_df = fetch_researchers()
    
    grants_df = extract_grant_features(grants_df)
    researchers_df = extract_researcher_features(researchers_df)
    
    X = np.array(researchers_df["concept_embeddings"].tolist())
    y = np.random.randint(2, size=len(researchers_df))
    group_ids = np.array([1] * len(researchers_df))
    
    model = LGBMRanker(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="lambdarank"
    )
    model.fit(X, y, group=group_ids)
    
    model.save_model("lightgbm_ranking_model.txt")
    print("LightGBM ranking model saved!")

if __name__ == "__main__":
    train_lightgbm_ranking_model()