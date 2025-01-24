# utils.py
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd
from loguru import logger

# Load environment variables
load_dotenv()

def connect_to_db():
    """Connect to the Supabase PostgreSQL database using SQLAlchemy."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not set")
        return None
    
    try:
        engine = create_engine(DATABASE_URL)
        logger.success("Successfully connected to the database")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def fetch_data_as_df(query, engine):
    """Fetch data from the database and return it as a pandas DataFrame."""
    if not engine:
        logger.error("No database connection provided")
        return None
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                logger.warning("Query returned no data")
            else:
                logger.success(f"Fetched {len(df)} rows")
            return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None