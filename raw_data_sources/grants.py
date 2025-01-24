from .utils import connect_to_db, fetch_data_as_df
from loguru import logger

def fetch_grants():
    """Fetch and clean grants data."""
    query = "SELECT * FROM grants_data;"  # Ensure table name is correct
    engine = connect_to_db()
    if not engine:
        logger.error("Database connection failed")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    grants_df = fetch_data_as_df(query, engine)
    if grants_df is None:
        logger.error("Failed to fetch grants data")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    # Basic cleaning
    if "summary_description" not in grants_df.columns:
        logger.error("Column 'summary_description' not found in grants data")
        return pd.DataFrame()
    
    grants_df = grants_df.dropna(subset=["summary_description"])  # Drop rows with missing descriptions
    if "award_ceiling" in grants_df.columns:
        grants_df["award_ceiling"] = grants_df["award_ceiling"].fillna(0)
    if "award_floor" in grants_df.columns:
        grants_df["award_floor"] = grants_df["award_floor"].fillna(0)
    
    logger.success(f"Cleaned grants data: {len(grants_df)} rows")
    return grants_df