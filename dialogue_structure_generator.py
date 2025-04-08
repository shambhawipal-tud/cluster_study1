import rdflib
import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print("tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print("model loaded")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model moved to GPU")
    return tokenizer, model, device

def get_llm_response(tokenizer, model, device, prompt):
   
    # Tokenize and get response from model
    inputs = tokenizer(text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Debug print to see the raw response
    print(f"Raw response: {response}")
    return response


# Function to parse dialogue file
def parse_dialogue(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    # Keep only dialogue lines (lines with ':')
    dialogues = [line for line in lines if ':' in line]
    print(dialogues)
    return dialogues


# Function to dynamically extract topic from an utterance
def extract_topic_from_utterance(utterance, tokenizer, model, device):
    prompt = f"Identify the main topic or subject of the following sentence: '{utterance}'"
    response = get_llm_response(tokenizer, model, device, prompt)
    
    # Extract topic from LLM response, assuming response is the topic
    topic = response.strip()  # In this case, we assume the model directly returns the topic
    return topic


# Function to extract claim from an utterance
def extract_claim_from_utterance(utterance, tokenizer, model, device):
    prompt = f"Is there any claim or assertion made in the following sentence: '{utterance}'? If yes, summarize it."
    response = get_llm_response(tokenizer, model, device, prompt)
    
    # Extract claim from LLM response
    claim = response.strip()  # Assume the LLM returns the claim
    return claim


# Function to create RDF triples for the dialogue
def create_rdf(dialogues, tokenizer, model, device):
    g = rdflib.Graph()
    EX = rdflib.Namespace("http://example.org/")
    g.bind("ex", EX)

    participants = {}  # Store participants

    for idx, (participant, utterance) in enumerate(dialogues):
        # Create participant nodes if not already created
        if participant not in participants:
            participants[participant] = EX[participant]

        # Create unique ID for each utterance (based on its index)
        utterance_node = EX[f"Utterance{idx+1}"]

        # Add triples for the participant and utterance
        g.add((participants[participant], rdflib.RDF.type, EX.Person))
        g.add((utterance_node, rdflib.RDF.type, EX.Utterance))
        g.add((utterance_node, EX.text, rdflib.Literal(utterance)))
        g.add((utterance_node, EX.speaker, participants[participant]))

        # Adding the role of the participant (e.g., Complainant, Accused, Facilitator)
        role = "Facilitator"  # Example, should be inferred from the context
        role_node = EX[role]
        g.add((participants[participant], EX.HAS_ROLE, role_node))

        # Dynamically extract claim from the utterance
        claim = extract_claim_from_utterance(utterance)
        if claim:
            claim_node = EX[claim.replace(" ", "_")]
            g.add((utterance_node, EX.MAKES_CLAIM, claim_node))

        # Dynamically extract topic from the utterance using LLM
        topic = extract_topic_from_utterance(utterance)
        if topic:
            topic_node = EX[topic.replace(" ", "_")]
            g.add((utterance_node, EX.MENTIONS_TOPIC, topic_node))

        # Dynamically extract stance based on the content of the utterance
        stance = "Support" if "agree" in utterance.lower() else "Neutral"
        stance_node = EX[stance]
        g.add((utterance_node, EX.HAS_STANCE, stance_node))

        # Dynamically extract intent (Clarify, Reassure, etc.)
        intent = "Clarify" if "clarify" in utterance.lower() else "Defend"
        intent_node = EX[intent]
        g.add((utterance_node, EX.HAS_INTENT, intent_node))

        # Check if this utterance refers to the previous one
        if idx > 0:
            prev_utterance_node = EX[f"Utterance{idx}"]
            g.add((utterance_node, EX.REFERS_TO, prev_utterance_node))

        # Check if evidence is provided (e.g., timestamp, document reference)
        if "performance" in utterance.lower():
            evidence = "Message sent at 11:07 PM"
            evidence_node = EX[evidence.replace(" ", "_")]
            g.add((utterance_node, EX.PROVIDES_EVIDENCE, evidence_node))

    return g


# Function to insert the RDF graph into GraphDB using SPARQL
def insert_into_graphdb(graph, sparql_endpoint):
    headers = {'Content-Type': 'application/x-turtle'}
    rdf_data = graph.serialize(format="turtle")

    response = requests.post(sparql_endpoint, data=rdf_data, headers=headers)

    if response.status_code == 200:
        print("Data inserted successfully")
    else:
        print(f"Failed to insert data. Status code: {response.status_code}")


# Main function
def main():
    file_path = r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt"  
    sparql_endpoint = 'http://tud1006635:7200/repositories/group_discussion' 
    tokenizer, model, device = load_llm()
    dialogues = parse_dialogue(file_path)
    rdf_graph = create_rdf(dialogues, tokenizer, model, device)
    insert_into_graphdb(rdf_graph, sparql_endpoint)


if __name__ == "__main__":
    main()
