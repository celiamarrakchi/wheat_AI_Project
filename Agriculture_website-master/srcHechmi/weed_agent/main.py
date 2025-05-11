from crew import CrewConfig, Crew  # Importation correcte de CrewConfig et Crew


def main():
    agents_path = "srcHechmi/weed_agent/config/agents.yaml"
    tasks_path = "srcHechmi/weed_agent/config/tasks.yaml"

    config = CrewConfig.from_yaml(agents_path, tasks_path)
    if config:
        crew = Crew(
            agents=config.agents,
            tasks=config.tasks,
            verbose=True
        )
        results = crew.kickoff()  # Assurez-vous qu'il n'y a qu'un seul appel à cette fonction
        print(f"📊 Résultats finaux: {results}")
    else:
        print("⚠️ Impossible de démarrer Crew en raison d'une erreur dans la configuration.")


if __name__ == "__main__":
    main()  # Appel de main()
