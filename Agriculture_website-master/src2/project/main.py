from crew import crew, config
from crew2 import crew, config
import os
def get_image_paths_from_folder(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png')
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_formats)
    ]

if __name__ == "__main__":

    # Lancer l'exécution des tâches avec l'agent
    result = crew.kickoff()
    print(result)
