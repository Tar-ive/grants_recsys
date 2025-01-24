# Grant Recommendation System

A machine learning system that matches researchers with relevant funding opportunities based on their profiles and grant details.

## Features
- Data fetching from Supabase PostgreSQL database
- Feature engineering with TF-IDF embeddings
- Two-Tower model using TensorFlow Recommenders
- CatBoost ranking model for grant recommendations
- Evaluation metrics (Precision@K, MAP)

## Project Structure
```
grant_recsys/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── raw_data_sources/      # Data fetching scripts
├── features/              # Feature engineering
├── training/              # Model training
├── test.py               # Testing script
└── README.md             # Documentation
```

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
Create `.env` file:
```bash
DATABASE_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:6543/postgres"
```

3. **Run Tests**
```bash
python test.py
```

4. **Train Models**
```bash
python training/two_tower.py
python training/ranking.py
```

## Models

### Two-Tower Model
- Neural networks for researcher and grant embeddings
- Uses TensorFlow Recommenders
- Output directory: `two_tower_model/`

### CatBoost Ranking
- Gradient boosting for grant ranking
- Uses YetiRank loss function
- Output: `catboost_ranking_model.cbm`

## Dependencies
- tensorflow-recommenders
- catboost
- loguru
- pandas
- scikit-learn
- sqlalchemy
- python-dotenv

## License
MIT License