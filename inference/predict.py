import tensorflow as tf

def load_model():
    return tf.keras.models.load_model("models/two_tower_model")

def predict(input_data):
    model = load_model()
    return model.predict(input_data)