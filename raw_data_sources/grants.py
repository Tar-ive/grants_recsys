from .utils import connect_to_db, fetch_data_as_df

# raw_data_sources/grants.py
def fetch_grants():
    """Fetch and clean grants data."""
    query = "SELECT * FROM grants_data;"
    engine = connect_to_db()
    if engine:
        grants_df = fetch_data_as_df(query, engine)
        # Basic cleaning
        grants_df = grants_df.dropna(subset=["summary_description"])  # Drop rows with missing descriptions
        grants_df["funding_categories"] = grants_df["funding_categories"].fillna("")  # Fill missing categories
        return grants_df
    return None