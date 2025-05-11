

from crew import crew

if __name__ == "__main__":
    # crew.kickoff() lira config/tasks.yaml et utilisera l'image uploadée
    result = crew.kickoff()
    print(result)