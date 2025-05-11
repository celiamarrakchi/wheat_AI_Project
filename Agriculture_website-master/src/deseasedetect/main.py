from crew import crew, config
import os

def get_image_paths_from_folder(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png')
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_formats)
    ]

if __name__ == "__main__":
    """# Récupérer les chemins d'images du dossier
    image_folder = "src/deseasedetect/images"
    image_paths = get_image_paths_from_folder(image_folder)

    # Injecter dynamiquement les chemins d’images dans les inputs de tâche
    for task in config.tasks.get("tasks", []):
        if "image_paths" in task.get("input", {}):
            task["input"]["image_paths"] = image_paths"""

    result = crew.kickoff()
    print(result)
