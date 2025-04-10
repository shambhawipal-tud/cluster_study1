import requests
import urllib.parse
from datetime import datetime
from openai import OpenAI

MODEL_NAME = "qwen2.5"
TEMPERATURE = 0.3

GRAPHDB_ENDPOINT = "http://127.0.0.1:7200/repositories/group_discussion/statements"
HEADERS = {
    "Content-Type": "application/sparql-update"
}


def sanitize_uri(text):
    """
    Shortens and sanitizes text to create a URI-safe slug.
    """
    short_text = text.split(".")[0][:100]  # Take first sentence only, max 100 chars
    sanitized = short_text.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "").replace(",", "").replace("'", "").replace(".", "")
    return urllib.parse.quote(sanitized)


def load_llm():
    client = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama"
    )
    return client


def get_llm_response(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def parse_dialogue(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    dialogues = []
    for line in lines:
        if ':' in line:
            participant, utterance = line.split(":", 1)
            dialogues.append((participant.strip(), utterance.strip()))
    return dialogues


def insert_data_to_graphdb(person_id, utterance_id, content, previous_utterance_uri, timestamp):
    person_uri = f"urn:person:{person_id}"
    utterance_uri = f"urn:utterance:{person_id}_{utterance_id}"

    sparql_update = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX urn: <urn:>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    INSERT DATA {{
        <{person_uri}> rdfs:label "Person {person_id}" .

        <{utterance_uri}> rdfs:label "{content}" .
        <{utterance_uri}> <urn:spokenBy> <{person_uri}> .
        <{utterance_uri}> <urn:hasTimestamp> "{timestamp}"^^xsd:dateTime .

        {f'<{utterance_uri}> <urn:refersTo> <{previous_utterance_uri}> .' if previous_utterance_uri else ""}
    }}
    """

    print("SPARQL Update:\n", sparql_update)
    try:
        response = requests.post(GRAPHDB_ENDPOINT, data=sparql_update, headers=HEADERS)
        response.raise_for_status()
        if response.status_code == 204:
            print(f"Inserted utterance {utterance_id} for {person_id}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception occurred: {e}")


def process_dialogue_file(file_path):
    client = load_llm()
    dialogues = parse_dialogue(file_path)

    previous_utterance_uri = None

    for idx, (participant, utterance) in enumerate(dialogues):
        person_id = sanitize_uri(participant)
        utterance_id = f"uttr{idx+1}"
        timestamp = datetime.utcnow().isoformat()

        # Insert data with connection to the previous utterance
        insert_data_to_graphdb(person_id, utterance_id, utterance, previous_utterance_uri, timestamp)

        # Update the previous_utterance_uri to the current one for the next round
        previous_utterance_uri = f"urn:utterance:{person_id}_{utterance_id}"


if __name__ == "__main__":
    process_dialogue_file(r"C:/Users/shambhawi/Source/Repos/cluster_study1/Case Study-1.txt")
