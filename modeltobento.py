"""Saves a Keras model to BentoML Store.

Guarda un modelo Keras en el BentoML Store

"""

from pathlib import Path
from tensorflow import keras
import bentoml


def model_to_bento(model_file: Path) -> None:
    """Loads a keras model from directory and saves it to BentoML Store."""
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("flower_model", model)
    print(f"Bento model tag = {bento_model.tag}")

if __name__ == "__main__":
    model_to_bento(Path('final_iris_model.h5'))