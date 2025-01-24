from .utils import connect_to_db, fetch_data_as_df

def fetch_researchers():
    """Fetch and clean researchers data."""
    query = "SELECT * FROM researchers;"
    engine = connect_to_db()
    if engine:
        researchers_df = fetch_data_as_df(query, engine)
        # Basic cleaning
        researchers_df = researchers_df.dropna(subset=["concept_1"])  # Drop rows with missing research concepts
        researchers_df["concept_2"] = researchers_df["concept_2"].fillna("")  # Fill missing concept_2
        return researchers_df
    return None