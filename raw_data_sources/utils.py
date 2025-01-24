from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def connect_to_db():
    """Connect to the Supabase PostgreSQL database using SQLAlchemy."""
    DATABASE_URL = os.getenv("DATABASE_URL")
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def fetch_data_as_df(query, engine):
    """Fetch data from the database and return it as a pandas DataFrame."""
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None