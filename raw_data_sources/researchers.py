# raw_data_sources/researchers.py
from .utils import connect_to_db, fetch_data_as_df
from loguru import logger

def fetch_researchers():
    """Fetch and clean researchers data."""
    query = "SELECT * FROM researchers;"
    engine = connect_to_db()
    if not engine:
        logger.error("Database connection failed")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    researchers_df = fetch_data_as_df(query, engine)
    if researchers_df is None:
        logger.error("Failed to fetch researchers data")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    # Basic cleaning
    if "concept_1" not in researchers_df.columns:
        logger.error("Column 'concept_1' not found in researchers data")
        return pd.DataFrame()
    
    researchers_df = researchers_df.dropna(subset=["concept_1"])  # Drop rows with missing research concepts
    if "concept_2" in researchers_df.columns:
        researchers_df["concept_2"] = researchers_df["concept_2"].fillna("")  # Fill missing concept_2
    
    logger.success(f"Cleaned researchers data: {len(researchers_df)} rows")
    return researchers_df