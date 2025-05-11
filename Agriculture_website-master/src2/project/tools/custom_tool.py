import os
import json
import numpy as np
from typing import Type
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from pathlib import Path

# ===== Classificateur de maladie du blé =====

# src2/project/wheatDiseaseModel.h5'


class HarvestDiseaseClassifier:
    def __init__(self, model_path=r'C:\Users\merye\Desktop\Agriculture_website-master\Agriculture_website-master\src2\project\wheatDiseaseModel.h5'):
        self.model = load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]
        self.class_names = [
            'Wheat_Loose_Smut',
            'Wheat_crown_root_rot',
            'Wheat_healthy'
        ]

    def preprocess(self, img_path):
        img = image.load_img(img_path, target_size=self.input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict(self, img_path):
        img_array = self.preprocess(img_path)
        predictions = self.model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = self.class_names[predicted_index]
        confidence = float(np.max(predictions)) * 100
        return f"{predicted_class} ({confidence:.2f}%)"

# ===== Input du Tool =====


class HarvestWheatDiseaseDetectionInput(BaseModel):
    image_path: str = Field(..., description="Chemin vers l'image du blé à analyser")

# ===== Tool CREWAI 1 =====


class HarvestWheatDiseaseDetectionTool(BaseTool):
    name: str = "Harvest Wheat Disease Detection Tool"
    description: str = "Détecte les maladies sur les blés à partir d'une image."
    args_schema: Type[BaseModel] = HarvestWheatDiseaseDetectionInput

    def _run(self, image_path: str) -> str:
        image_path = Path(image_path).as_posix()

        classifier = HarvestDiseaseClassifier()
        result = classifier.predict(image_path)
        print(f"🔬 Maladie détectée : {result}")
        self.save_result({"image": image_path, "disease": result})
        return result

    def save_result(self, data):
        try:
            file_path = "harvest_wheat_disease_results.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(data)

            with open(file_path, "w") as f:
                json.dump(existing, f, indent=4)
            print("✅ Résultat enregistré.")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")

# ===== Tool CREWAI 2 =====


class solutionWebsiteSearchTool:
    def search(self, disease_name, image_path=None):
        query = f"{disease_name} treatment for wheat disease"
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            for result in results:
                solution = result['body']
                if (disease_name == "Wheat_healthy"):
                    solution = "No treatment needed. The wheat is healthy"

                self.save_result({
                    "image": image_path if image_path else "N/A",
                    "disease": disease_name,
                    "solution": solution
                })
                return solution
        return "No treatment information found."

    def save_result(self, data):
        try:
            file_path = "harvest_wheat_disease_results.json"
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)
            print("✅ Info sauvegardée pour :", data["disease"])
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde dans solutionWebsiteSearchTool : {e}")
