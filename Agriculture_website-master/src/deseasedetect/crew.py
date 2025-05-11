import yaml
from src.deseasedetect.tools.custom_tool import YoloPredictionTool,WheatDiseaseDetectionTool,WebsiteSearchTool

class CrewConfig:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks

    @classmethod
    def from_yaml(cls, agents_path, tasks_path):
        with open(agents_path, 'r') as a:
            agents = yaml.load(a, Loader=yaml.FullLoader)
        with open(tasks_path, 'r') as t:
            tasks = yaml.load(t, Loader=yaml.FullLoader)
        return cls(agents, tasks)

class Crew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents = agents
        self.tasks = tasks.get('tasks', [])
        self.verbose = verbose
        self.outputs = {}

    def kickoff(self):
        self.execute_tasks()
        return self.outputs

    def resolve_input(self, input_data):
        if isinstance(input_data, dict):
            resolved = {}
            for key, value in input_data.items():
                if isinstance(value, str) and value.startswith("{{") and "output" in value:
                    parts = value.strip("{{}}").split(".")
                    task_name = parts[0]
                    output_key = parts[-1]
                    resolved[key] = self.outputs.get(task_name, {}).get(output_key, "UNKNOWN")
                else:
                    resolved[key] = value
            return resolved
        return input_data

    def execute_tasks(self):
        if self.verbose:
            print("📋 Agents chargés :", list(self.agents.keys()))
            print("📋 Tâches à exécuter :", [task['name'] for task in self.tasks])

        for task in self.tasks:
            name = task['name']
            agent_name = task['agent']
            tools = task.get("tools", [])
            input_data = self.resolve_input(task.get("input", {}))
            output = {}

            print(f"\n🚀 Exécution de la tâche: {name}")
            print(f"👤 Agent assigné: {agent_name}")
            print(f"🧰 Outils: {tools}")
            print(f"📥 Données d'entrée: {input_data}")

            for tool in tools:
                if tool == "DiseaseClassifier":
                    classifier = WheatDiseaseDetectionTool()
                    predictions = []
                    image_paths = input_data.get("image_paths", [])
                    for img_path in image_paths:
                        result = classifier._run(image_path=img_path)
                        predictions.append({"image": img_path, "disease": result})
                    output = {"predictions": predictions}
                elif tool == "WebsiteSearchTool":
                    searcher = WebsiteSearchTool()
                    previous_results = self.outputs.get("classify_disease_task", {}).get("predictions", [])
                    enriched_results = []
                    for prediction in previous_results:
                        disease = prediction["disease"]
                        image = prediction.get("image", "N/A")
                        info = searcher.search(disease, image)
                        prediction["info"] = info
                        enriched_results.append(prediction)
                    output = {"predictions": predictions}
                if "YoloPredictionTool" in tools:
                    yolo_tool = YoloPredictionTool()
                    outputs = []
                    for image_path in input_data["image_paths"]:
                        annotated_image, predictions = yolo_tool.predict_and_annotate(image_path=image_path)
                        outputs.append({
                            "image_path": image_path,
                            "annotated_image": str(annotated_image),
                            "predictions": predictions
                        })
                    

            self.outputs[name] = output
            print(f"✅ Résultat de la tâche {name}: {output}")
            print("-" * 50)



config = CrewConfig.from_yaml(
    agents_path="src/deseasedetect/config/agents.yaml",
    tasks_path="src/deseasedetect/config/tasks.yaml"
)

crew = Crew(
    agents=config.agents,
    tasks=config.tasks,
    verbose=True
)

