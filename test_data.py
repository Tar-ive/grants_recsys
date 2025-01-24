from raw_data_sources.grants import fetch_grants
from raw_data_sources.researchers import fetch_researchers
from features.grants import extract_grant_features
from features.researchers import extract_researcher_features

def main():
    # Fetch and clean data
    print("Fetching grants data...")
    grants_df = fetch_grants()
    if grants_df is not None:
        print(f"Fetched {len(grants_df)} grants!")
        grants_df = extract_grant_features(grants_df)  # Feature engineering
        print(grants_df[["opportunity_id", "description_embeddings"]].head())

    # Fetch and clean researchers data
    print("\nFetching researchers data...")
    researchers_df = fetch_researchers()
    if researchers_df is not None:
        print(f"Fetched {len(researchers_df)} researchers!")
        researchers_df = extract_researcher_features(researchers_df)  # Feature engineering
        print(researchers_df[["researcher_id", "concept_embeddings"]].head())

if __name__ == "__main__":
    main()