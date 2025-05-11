import os
import json
from typing import Type
from pydantic import BaseModel, Field
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from crewai.tools import BaseTool
import glob
from datetime import datetime
import shutil
from duckduckgo_search import DDGS

# Get the absolute path of the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_images_directory(images_dir):
    """Completely cleans the images folder."""
    try:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            return

        for file in glob.glob(os.path.join(images_dir, "*.*")):
            try:
                os.remove(file)
                print(f"🗑️ File deleted: {file}")
            except Exception as e:
                print(f"⚠️ Error while deleting file {file}: {e}")
        
        print(f"✅ Image folder cleaned: {images_dir}")
    except Exception as e:
        print(f"⚠️ Error while cleaning the image folder: {e}")

def save_uploaded_image(uploaded_file, images_dir):
    """Saves the uploaded image with a unique filename."""
    try:
        os.makedirs(images_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(uploaded_file.name)[1]
        filename = f"{timestamp}{file_extension}"
        file_path = os.path.join(images_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print(f"✅ Image saved: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Error while saving image: {e}")
        return None

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Query to search for weed management information")

class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = "Search for weed management information on reliable agricultural websites"
    args_schema: Type[BaseModel] = WebSearchInput

    def search(self, weed_status, image_path=None):
        query = f"{weed_status} management solutions for agricultural fields"
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            for result in results:
                solution = result['body']
                if weed_status == "No weed detected":
                    solution = "No treatment needed. The field is weed-free."

                self.save_result({
                    "image": image_path if image_path else "N/A",
                    "weed_status": weed_status,
                    "solution": solution
                })
                return solution
        return "No weed management information found."

    def save_result(self, data):
        try:
            results_dir = os.path.join(ROOT_DIR, "results")
            os.makedirs(results_dir, exist_ok=True)
            file_path = os.path.join(results_dir, "weed_results.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)
            print("✅ Info saved for:", data["weed_status"])
        except Exception as e:
            print(f"❌ Error while saving in WebSearchTool: {e}")

    def _run(self, query: str) -> str:
        try:
            return self.search(query)
        except Exception as e:
            return f"Error during web search: {str(e)}"

class WeedClassifierInput(BaseModel):
    image_path: str = Field(..., description="Path to the image to classify")

class WeedClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(ROOT_DIR, "models", "final_resnet50_l2_augmented.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, img_path):
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")

            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"❌ Error loading image: {e}")
            return -1

        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            prediction = (output.item() > 0.5)
        return int(prediction)

class WeedClassifierTool(BaseTool):
    name: str = "WeedClassifierTool"
    description: str = "Classifies an image as weed or not"
    args_schema: Type[BaseModel] = WeedClassifierInput

    def search_solution(self, weed_status, image_path=None):
        if weed_status == "No weed detected":
            solution = "No treatment necessary. Your field is healthy and weed-free."
        else:
            query = "how to remove weeds from agricultural fields practical methods"
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                solutions = []
                for result in results:
                    text = result['body'].replace('\n', ' ').strip()
                    if len(text) > 50:
                        solutions.append(text)
                
                if solutions:
                    solution = "Recommendations based on best agricultural practices:\n\n"
                    for i, sol in enumerate(solutions[:3], 1):
                        solution += f"{i}. {sol}\n\n"
                    solution += "Tip: These recommendations are based on reliable sources. Adapt them to your specific situation."
                else:
                    solution = "Sorry, no specific recommendations could be found at the moment. Please consult a local agricultural expert."

        self.save_result({
            "image": image_path if image_path else "N/A",
            "weed_status": weed_status,
            "solution": solution
        })
        return solution

    def save_result(self, data):
        try:
            results_dir = os.path.join(ROOT_DIR, "results")
            os.makedirs(results_dir, exist_ok=True)
            file_path = os.path.join(results_dir, "weed_results.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)
            print("✅ Info saved for:", data["weed_status"])
        except Exception as e:
            print(f"❌ Error while saving: {e}")

    def _run(self, image_path: str) -> str:
        try:
            model = WeedClassifier()
            result = model.predict(image_path)

            if result == -1:
                return json.dumps({
                    "error": "Error while processing the image.",
                    "image": image_path
                })

            weed_status = "Weed detected" if result == 1 else "No weed detected"
            solution = self.search_solution(weed_status, image_path)

            return json.dumps({
                "image": image_path,
                "prediction": result,
                "message": f"Predicted class: {result} (0 = not weed, 1 = weed)",
                "solution": solution
            })
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "image": image_path
            })
