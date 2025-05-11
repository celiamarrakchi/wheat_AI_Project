import os
from pathlib import Path
from crew import crew, config

# Function to gather all supported image file paths from a folder
def get_image_paths_from_folder(folder_path):
    supported_formats = ('.jpg', '.jpeg', '.png')
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(supported_formats)
    ]

# Define a placeholder config object for demonstration
class Config:
    def __init__(self):
        self.tasks = [
            {
                'input': {
                    'image_paths': []  # This will be dynamically populated later
                }
            }
        ]

if __name__ == "__main__":
    # Get the path to the project directory (three levels up from src/project)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define the absolute path to the images folder
    image_folder = os.path.join(project_dir, 'src', 'project', 'images')  # Relative to the project root
    
    # Get image paths from the folder for insect detection
    image_paths = get_image_paths_from_folder(image_folder)

    # Print the absolute image paths for debugging purposes
    print("Using these images:", image_paths)

    # Initialize the config object
    config = Config()

    # Update the task configuration with the image paths
    for task in config.tasks:
        if "input" in task and "image_paths" in task["input"]:
            task["input"]["image_paths"] = image_paths

    # Print the updated config for debugging
    print("Updated config:", config.tasks)

    # Assuming `crew` is already defined and initialized elsewhere in your code
    result = crew.kickoff()
    print(result)