import os
import numpy as np
import cv2
from dotenv import load_dotenv
from typing import Type, ClassVar
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import json

from tools.utils import send_email_notification

# Charger les variables d'environnement
load_dotenv()

# ===== Fonction pour sauvegarder les résultats dans un fichier JSON =====
def save_results(file_path, results):
    """Sauvegarde les résultats dans un fichier JSON"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                existing_results = json.load(file)
        else:
            existing_results = []

        existing_results.append(results)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(existing_results, file, ensure_ascii=False, indent=4)

        print(f"✅ Résultats enregistrés dans {file_path}")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement des résultats: {e}")

# ===== Classe LLM personnalisée =====
class LLMToolGroq:
    def __init__(self, api_key=None, model_name="llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.stop = []

    def supports_stop_words(self) -> bool:
        return False

    def predict(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error while executing the task: {str(e)}"

    def call(self, prompt: str) -> str:
        return self.predict(prompt)

# ===== Classe prédicteur d'image =====
class WheatGrowthPredictor:
    def __init__(self, model_path="model_inceptionv3.h5"):
        self.model = load_model(model_path)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array([
            'Filling',
            'Filling Ripening',
            'Post flowering',
            'Ripening'
        ])

    def predict(self, image_path: str) -> str:
        # Normalise le chemin et log pour debug
        image_path = os.path.normpath(image_path)
        print(f"🔍 Trying to load image at: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image from: {image_path}")
            return "Erreur : Image introuvable."

        # Pré-traitement
        image = cv2.resize(image, (299, 299)) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prédiction
        predictions = self.model.predict(image)
        predicted_index = np.argmax(predictions, axis=1)[0]
        if predicted_index >= len(self.label_encoder.classes_):
            predicted_index = len(self.label_encoder.classes_) - 1
        stage = self.label_encoder.inverse_transform([predicted_index])[0]

        print(f"✅ Loaded and predicted stage: {stage}")
        if stage == "Ripening":
            send_email_notification("masmoudi.emna.1@esprit.tn", stage)

        return stage

# ===== Tool 1: Prédiction du stade =====
class WheatGrowthInput(BaseModel):
    image_path: str = Field(..., description="Path to the wheat image to analyze")

class WheatGrowthPredictionTool(BaseTool):
    name: str = "Wheat Growth Stage Prediction Tool"
    description: str = "Predicts the growth stage of wheat and provides advice."
    args_schema: Type[BaseModel] = WheatGrowthInput
    llm: ClassVar[LLMToolGroq] = LLMToolGroq()

    def _run(self, image_path: str) -> dict:
        predictor = WheatGrowthPredictor(model_path="model_inceptionv3.h5")
        growth_stage = predictor.predict(image_path)
        print(f"🔍 Predicted growth stage: {growth_stage}")
        return {'growth_stage': growth_stage}

# ===== Tool 2: Conseils agricoles =====
class AdviceInput(BaseModel):
    growth_stage: str = Field(..., description="The growth stage of the wheat")

class WheatAdviceTool(BaseTool):
    name: str = "Wheat Growth Advice Tool"
    description: str = "Provides farming advice based on the growth stage of wheat."
    args_schema: Type[BaseModel] = AdviceInput
    llm: ClassVar[LLMToolGroq] = LLMToolGroq()

    def _run(self, growth_stage: str) -> str:
        prompt = f"You are an agriculture expert. Provide practical advice to a farmer to optimize wheat cultivation at the '{growth_stage}' growth stage."
        advice = self.llm.call(prompt)
        save_results("results.json", {"growth_stage": growth_stage, "advice": advice})
        return advice
