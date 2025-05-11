import yaml

import yaml

class CrewConfig:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks

    @classmethod
    def from_yaml(cls, agents_path, tasks_path):
        # Charger les agents depuis un fichier YAML
        with open(agents_path, 'r') as agents_file:
            agents = yaml.load(agents_file, Loader=yaml.FullLoader)

        # Charger les tâches depuis un fichier YAML
        with open(tasks_path, 'r') as tasks_file:
            tasks = yaml.load(tasks_file, Loader=yaml.FullLoader)

        return cls(agents, tasks)


from tools.custom_tools import WheatGrowthPredictionTool, WheatAdviceTool

class Crew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.outputs = {}  # Pour stocker les sorties des tâches

    def kickoff(self):
        self.execute_tasks()
        return self.outputs

    def resolve_input(self, input_data):
        # Remplace les variables dynamiques comme {{PredictWheatStage.output.growth_stage}}
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
            print(f"Exécution des tâches avec les agents : {self.agents}")

        tasks = self.tasks.get('tasks', [])
        for task in tasks:
            name = task['name']
            tools = task.get("tools", [])
            input_data = self.resolve_input(task.get("input", {}))
            output = {}

            print(f"Exécution de la tâche: {name}")
            print(f"Entrées de la tâche: {input_data}")

            for tool in tools:
                if tool == "Wheat Growth Stage Prediction Tool":
                    tool_instance = WheatGrowthPredictionTool()
                    output = tool_instance._run(**input_data)
                elif tool == "Wheat Growth Advice Tool":
                    tool_instance = WheatAdviceTool()
                    output = {"advice": tool_instance._run(**input_data)}

            self.outputs[name] = output
            print(f"✅ Résultat de la tâche {name}: {output}")
            print("-" * 40)



# Modifier les chemins pour qu'ils pointent vers le fichier correct
config = CrewConfig.from_yaml(
    agents_path=r"C:\Users\MSI\Documents\Agriculture_website-master\Agriculture_website-master\Agriculture_website-master\src1\new_project\config\agent.yaml",  # a changer
    tasks_path=r"C:\Users\MSI\Documents\Agriculture_website-master\Agriculture_website-master\Agriculture_website-master\src1\new_project\config\tasks.yaml"   # a changer
)

# Créer la Crew et exécuter les tâches
crew = Crew(
    agents=config.agents,
    tasks=config.tasks,
    verbose=True
)

