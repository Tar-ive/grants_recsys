import tensorflow as tf
import tensorflow_recommenders as tfrs
from loguru import logger
from features.grants import extract_grant_features
from features.researchers import extract_researcher_features
from raw_data_sources.grants import fetch_grants
from raw_data_sources.researchers import fetch_researchers

# Loguru setup
logger.add("two_tower.log", rotation="10 MB", level="INFO")

class TwoTowerModel(tfrs.Model):
    def __init__(self, researcher_embedding_dim, grant_embedding_dim):
        super().__init__()
        # Researcher Tower
        self.researcher_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(researcher_embedding_dim),
        ])
        
        # Grant Tower
        self.grant_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(grant_embedding_dim),
        ])
        
        # Task: Predict similarity between researcher and grant
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(grant_embeddings).batch(128).map(self.grant_embedding)
            )
        )

    def compute_loss(self, features, training=False):
        researcher_embeddings = self.researcher_embedding(features["researcher_features"])
        grant_embeddings = self.grant_embedding(features["grant_features"])
        
        return self.task(researcher_embeddings, grant_embeddings)

def train_two_tower_model():
    logger.info("Fetching and preprocessing data...")
    grants_df = fetch_grants()
    researchers_df = fetch_researchers()
    
    # Extract features
    grants_df = extract_grant_features(grants_df)
    researchers_df = extract_researcher_features(researchers_df)
    
    # Prepare data for TFRS
    researcher_embeddings = np.array(researchers_df["concept_embeddings"].tolist())
    grant_embeddings = np.array(grants_df["description_embeddings"].tolist())
    
    # Create TensorFlow datasets
    researcher_dataset = tf.data.Dataset.from_tensor_slices({
        "researcher_features": researcher_embeddings
    })
    grant_dataset = tf.data.Dataset.from_tensor_slices({
        "grant_features": grant_embeddings
    })
    
    # Train the model
    logger.info("Training two-tower model...")
    model = TwoTowerModel(researcher_embedding_dim=64, grant_embedding_dim=64)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    # Train
    model.fit(researcher_dataset.batch(128), epochs=10)
    logger.info("Training complete!")
    
    # Save the model
    model.save("two_tower_model")
    logger.info("Model saved to 'two_tower_model'.")

if __name__ == "__main__":
    train_two_tower_model()