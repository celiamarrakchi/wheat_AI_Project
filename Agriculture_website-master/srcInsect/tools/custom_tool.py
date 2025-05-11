import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path
from pydantic import BaseModel
from typing import Type
from crewai.tools import BaseTool
import json
import os
import requests  # For web search

class InsectClassifier:
    def __init__(self, model_path=None):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Updated with actual insect names from IP102 dataset
        self._classes = [
            "Brown planthopper",  # class_25
            "Small brown planthopper",  # class_28
            "White-backed planthopper",  # class_29
            "Rice leafroller",  # class_30
            "Rice leaf caterpillar",  # class_31
            "Paddy stem maggot",  # class_32
            "Asiatic rice borer",  # class_33
            "Rice stem fly",  # class_34
            "Striped stem borer",  # class_35
            "Black bug",  # class_37
            "Rice water weevil",  # class_38
            "Rice hispa",  # class_40
            "Rice grasshopper",  # class_41
            "Rice skipper",  # class_42
            "Rice shell pest",  # class_43
            "Rice thrips",  # class_54
            "Rice gall midge",  # class_55
            "Rice stem maggot"  # class_57
        ]
        
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / 'resnet101_finetuned_final.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path.resolve()}")
        
        self._model_path = model_path
        self._model = self._load_model()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, len(self._classes))
        )
        model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        model.to(self._device)
        model.eval()
        return model

    def predict(self, img_path):
        img_path = Path(img_path) if not isinstance(img_path, Path) else img_path
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found at {img_path.resolve()}")
            
        img = Image.open(img_path).convert('RGB')
        img_tensor = self._transform(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prob = torch.nn.functional.softmax(outputs, dim=1)[0]

        class_name = self._classes[predicted.item()]
        confidence = prob[predicted.item()].item() * 100

        return class_name, confidence

class InsectClassificationInput(BaseModel):
    image_path: str

class WebSearchInput(BaseModel):
    insect_name: str

class InsectClassifierTool(BaseTool):
    name: str = "InsectClassifier"
    description: str = "Classifies insects in wheat plant images using a ResNet101 model."
    args_schema: Type[BaseModel] = InsectClassificationInput

    def _run(self, image_path: str) -> str:
        if not os.path.isabs(image_path):
            images_dir = Path(__file__).resolve().parent.parent / 'images'
            image_path = str(images_dir / image_path)
        
        classifier = InsectClassifier()
        try:
            class_name, confidence = classifier.predict(image_path)
            result = f"Insect classified as: {class_name} (Confidence: {confidence:.2f}%)"
            print(f"🔬 Classification result: {result}")
            self.save_result({"image": image_path, "classification": result})
            return result
        except Exception as e:
            error_msg = f"Error processing image {image_path}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def save_result(self, data):
        try:
            project_root = Path(__file__).resolve().parents[2]
            file_path = project_root / "insect_classification_results.json"

            if file_path.exists():
                with open(file_path, "r") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(data)

            with open(file_path, "w") as f:
                json.dump(existing, f, indent=4)
            print("✅ Classification result saved.")
        except Exception as e:
            print(f"❌ Error saving result: {e}")

class WebsiteSearchTool(BaseTool):
    name: str = "WebsiteSearchTool"
    description: str = "Performs web search for insect-related information."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, insect_name: str) -> str:
        search_result = self.perform_web_search(insect_name)
        return search_result

    def perform_web_search(self, insect_name: str):
        # Example search, replace this with actual API calls or scraping
        search_results = f"Web search result for {insect_name}: Detailed info from web search."
        return search_results

# Add this function to fix the import error
def recommend_insect_info(insect_name: str) -> str:
    """
    Provides information about the identified insect.
    
    Args:
        insect_name (str): The name of the insect detected in the image
        
    Returns:
        str: Information about the insect, potential threats to wheat crops, and recommended treatments
    """
    # Dictionary of common wheat insect pests and their information
    insect_info = {
        "Brown planthopper": {
            "description": "Small brownish insect that feeds on the phloem sap of rice plants.",
            "threat": "Can cause 'hopper burn' where leaves turn yellow and dry up. Significant yield losses in severe infestations.",
            "treatment": "Use resistant varieties, balanced fertilization, and appropriate insecticides like Imidacloprid or Buprofezin."
        },
        "Small brown planthopper": {
            "description": "Smaller variant of planthopper that damages rice plants by sucking sap.",
            "threat": "Vectors rice stripe virus. Can lead to stunted growth and reduced yield.",
            "treatment": "Maintain field cleanliness, use resistant varieties, and apply neem-based insecticides."
        },
        "White-backed planthopper": {
            "description": "Distinguished by its white back, this planthopper feeds on plant sap.",
            "threat": "Causes wilting, reduced growth, and can transmit viral diseases.",
            "treatment": "Encourage natural enemies, use resistant varieties, and apply targeted insecticides."
        },
        "Rice leafroller": {
            "description": "Caterpillar that rolls rice leaves and feeds within the rolled leaf.",
            "threat": "Damages photosynthetic area leading to reduced grain filling and yield.",
            "treatment": "Early planting, remove egg masses, use Trichogramma parasitoids or Bacillus thuringiensis sprays."
        },
        "Rice leaf caterpillar": {
            "description": "Larva that feeds on rice leaves, creating characteristic damage patterns.",
            "threat": "Reduces photosynthetic area and plant vigor, affecting yield.",
            "treatment": "Maintain field hygiene, encourage natural enemies, and use selective insecticides."
        }
    }
    
    # Extract just the insect name from potential classification string
    if ":" in insect_name:
        insect_name = insect_name.split(":")[1].strip()
        if "(" in insect_name:
            insect_name = insect_name.split("(")[0].strip()
    
    # Clean up the insect name for matching
    for key in insect_info.keys():
        if key.lower() in insect_name.lower():
            insect_name = key
            break
    
    # Return information about the insect if available, otherwise a generic message
    if insect_name in insect_info:
        info = insect_info[insect_name]
        return f"""
        {insect_name}
        
        Description: {info['description']}
        
        Threat to Crops: {info['threat']}
        
        Recommended Treatment: {info['treatment']}
        """
    else:
        # For insects not in our database, provide a generic response
        return f"""
        Information for {insect_name} is currently limited in our database. 
        
        Generally, it's recommended to:
        1. Monitor crop fields regularly for signs of infestation
        2. Consider biological control methods first
        3. Consult with local agricultural extension services for specific treatment recommendations
        4. Apply targeted pesticides only when necessary and according to guidelines
        """