import os
import json
import numpy as np
from typing import Type
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from crewai.tools import base_tool
from ultralytics import YOLO
from pathlib import Path
import cv2

# ===== Classificateur de maladie du blé =====
class DiseaseClassifier:
    def __init__(self, model_path='src/deseasedetect/cnn_model.h5'):
        self.model = load_model(model_path)
        self.class_names = [
            "Wheat Brown-rust",
            "Wheat Healthy",
            "Wheat Smut",
            "Wheat-Yellow-rust",
            "Wheat Stem Rust"
        ]  # adapte selon tes classes

    def preprocess(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict(self, img_path):
        img_array = self.preprocess(img_path)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        if predicted_class >= len(self.class_names):
            predicted_class = len(self.class_names) - 1

        return self.class_names[predicted_class]


# ===== Input du Tool =====
class WheatDiseaseDetectionInput(BaseModel):
    image_path: str = Field(..., description="Chemin vers l'image du blé à analyser")


# ===== Tool CREWAI 1 =====
class WheatDiseaseDetectionTool(BaseTool):
    name: str = "Wheat Disease Detection Tool"
    description: str = "Détecte les maladies sur les feuilles de blé à partir d'une image."
    args_schema: Type[BaseModel] = WheatDiseaseDetectionInput

    def _run(self, image_path: str) -> str:
        classifier = DiseaseClassifier()
        result = classifier.predict(image_path)
        print(f"🔬 Maladie détectée : {result}")
        self.save_result({"image": image_path, "disease": result})
        return result

    def save_result(self, data):
        try:
            file_path = "wheat_disease_results.json"
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
class WebsiteSearchTool:
    def search(self, disease_name, image_path=None):
        query = f"{disease_name}"
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            for result in results:
                info = result['body']
                self.save_result({
                    "image": image_path if image_path else "N/A",
                    "disease": disease_name,
                    "info": info
                })
                return info
        return "No info found online."

    def save_result(self, data):
        try:
            file_path = "wheat_disease_results.json"
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
            print(f"❌ Erreur lors de la sauvegarde dans WebsiteSearchTool : {e}")
#Yolo
class YoloPredictionTool:
    def __init__(self, model_path='best.pt', output_dir="results"):
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def predict_and_annotate(self, image_path):
        predictions = self.predict(image_path)
        annotated_image_path = self.annotate_image(image_path, predictions)
        self.save_results(image_path, predictions)
        return annotated_image_path, predictions

    def predict(self, image_path):
        results = self.model(image_path)  # prédiction sur image
        predictions = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = self.model.names[int(box.cls[0])]
                predictions.append({
                    'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'label': label
                })

        return predictions

    def annotate_image(self, image_path, predictions):
        image = cv2.imread(image_path)

        for pred in predictions:
            x, y, w, h = pred['box']
            label = pred['label']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        annotated_image_path = self.output_dir / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(annotated_image_path), image)
        return annotated_image_path

    def save_results(self, image_path, predictions):
        results = []
        for pred in predictions:
            result = {
                "image": Path(image_path).name,
                "box": pred['box'],
                "label": pred['label']
            }
            results.append(result)

        json_file = self.output_dir / "predictions.json"
        with open(json_file, 'a') as f:
            json.dump(results, f, indent=4)