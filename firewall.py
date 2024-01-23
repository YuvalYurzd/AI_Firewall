from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModelForSequenceClassification, \
    pipeline
import torch
import re
import requests


def check_toxicity(prompt, api_key):
    perspective_api_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "comment": {"text": prompt},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}},
        "doNotStore": True,
    }

    params = {"key": api_key}

    response = requests.post(perspective_api_url, headers=headers, json=data, params=params)

    if response.status_code == 200:
        result = response.json()
        toxicity_score = result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return toxicity_score
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def detect_personal_info(prompt):
    # Define regular expressions for common personal information patterns
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
        'ssn': r'\b\d{3}[-]\d{2}[-]\d{4}\b',
        'address': r'\b\d+\s\w+\s\w+|\w+\s\d+\b',

    }

    personal_info = {}

    # Search for patterns in the prompt
    for key, pattern in patterns.items():
        matches = re.findall(pattern, prompt)
        if matches:
            personal_info[key] = matches

    return personal_info


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("laiyer/deberta-v3-base-prompt-injection")
    model = AutoModelForSequenceClassification.from_pretrained("laiyer/deberta-v3-base-prompt-injection")
    api_key = "AIzaSyDMOFboEyYtxsIK1jrh-KyeM4uSpKej0do"

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # "CPU" corrected to "cpu"
    )
    prompt = input("Enter prompt: ")
    toxicity_score = check_toxicity(prompt, api_key)
    result = classifier(prompt)
    label = result[0]['label']
    result = detect_personal_info(prompt)
    if label == 'INJECTION' or result or toxicity_score > 0.5:
        if label == 'INJECTION':
            print("Prompt injection detected")
        if result:
            print("Personal info detected")
        if toxicity_score > 0.5:
            print("Toxic prompt detected")
    else:
        print("Prompt is fine")
