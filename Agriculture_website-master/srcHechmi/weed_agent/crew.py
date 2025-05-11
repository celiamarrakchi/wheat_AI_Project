import yaml
import json
import os
from srcHechmi.weed_agent.tools.custom_tool import WeedClassifierTool

# Obtenir le chemin absolu du répertoire courant
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class CrewConfig:
    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks

    @classmethod
    def from_yaml(cls, agents_path, tasks_path):
        try:
            # Utiliser les chemins absolus
            agents_path = os.path.join(CURRENT_DIR, 'config', 'agents.yaml')
            tasks_path = os.path.join(CURRENT_DIR, 'config', 'tasks.yaml')
            
            with open(agents_path, 'r', encoding='utf-8') as a:
                agents = yaml.load(a, Loader=yaml.FullLoader)
            with open(tasks_path, 'r', encoding='utf-8') as t:
                tasks = yaml.load(t, Loader=yaml.FullLoader)
            return cls(agents, tasks)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement des fichiers YAML : {e}")
            return None


class Crew:
    def __init__(self, agents, tasks, verbose=False):
        self.agents = agents
        self.tasks = tasks.get('tasks', [])
        self.verbose = verbose  # Ajout de la gestion du mode verbose
        self.outputs = {}

    def kickoff(self):
        """Exécute les tâches et renvoie les résultats."""
        self.execute_tasks()
        return self.outputs

    def resolve_input(self, input_data):
        """Résout les variables dynamiques dans les données d'entrée (ex: {{ task.output }})"""
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
                if tool == "WeedClassifierTool":
                    classifier = WeedClassifierTool()
                    image_paths = input_data.get("image_paths")

                    if isinstance(image_paths, list):  # Vérifie si c'est une liste
                        results = []
                        for image_path in image_paths:
                            result_json = classifier._run(image_path=image_path)
                            try:
                                result_dict = json.loads(result_json)
                                results.append(result_dict)
                            except json.JSONDecodeError:
                                results.append({"error": "Résultat de classification illisible"})
                        output = results
                    else:  # Cas où image_paths est une seule chaîne
                        result_json = classifier._run(image_path=image_paths)
                        try:
                            result_dict = json.loads(result_json)
                            output = result_dict
                        except json.JSONDecodeError:
                            output = {"error": "Résultat de classification illisible"}

            self.outputs[name] = output
            print(f"✅ Résultat de la tâche {name}: {output}")
            print("-" * 50)


# Chargement de la configuration avec gestion des erreurs
config = CrewConfig.from_yaml(
    agents_path=os.path.join(CURRENT_DIR, 'config', 'agents.yaml'),
    tasks_path=os.path.join(CURRENT_DIR, 'config', 'tasks.yaml')
)

# Si la configuration est valide, démarrer le Crew
if config:
    crew = Crew(
        agents=config.agents,
        tasks=config.tasks,
        verbose=True
    )
else:
    print("⚠️ Impossible de démarrer Crew en raison d'une erreur dans la configuration.")
    crew = None

# Exportation explicite des objets
__all__ = ['crew', 'config']
