import json
import re
from sentence_transformers import SentenceTransformer
import chromadb

JSONL_PATH = "time_series_generation.jsonl"
CHROMA_DB_PATH = "./memoria_maestro"
COLLECTION_NAME = "time_series_generation"

patron_lista = r'\[\s*(?:-?\d+(?:\.\d+)?|\'[\dA-Za-z.-]+\'|"[\dA-Za-z.-]+")(?:\s*,\s*(?:-?\d+(?:\.\d+)?|\'[\dA-Za-z.-]+\'|"[\dA-Za-z.-]+"))*\s*\]'

# Modelo con 512 tokens 
model = SentenceTransformer('all-mpnet-base-v2')

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass
collection = client.create_collection(name=COLLECTION_NAME)

def extract_all_time_series(text):
    matches = re.findall(patron_lista, text)
    series_list = []
    for match in matches:
        inner = match[1:-1]
        values = []
        for token in inner.split(','):
            token = token.strip().strip("'\"")
            try:
                values.append(float(token))
            except ValueError:
                values.append(token)
        series_list.append(values)
    return series_list

def clean_description(text):
    text = re.sub(patron_lista, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_search_text(entry):
    domain = entry['metadata']['original_domain']
    task = entry['metadata']['task_type']
    response = entry['output']
    question = clean_description(entry['input'])
    
    text = f"Dominio: {domain}. Tarea: {task}. Pregunta: {question}. Respuesta: {response}."
    return text

with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        
        series_list_input = extract_all_time_series(entry['input'])
        series_list_output = extract_all_time_series(entry['output'])  # nombre corregido

        series = {
            "input": [{"longitud_puntos": len(x), "serie": x} for x in series_list_input],
            "output": [{"longitud_puntos": len(x), "serie": x} for x in series_list_output]
        }
        series_json = json.dumps(series)

        search_text = build_search_text(entry)
        embedding = model.encode(search_text).tolist()
        
        collection.add(
            ids=[entry['id']],
            embeddings=[embedding],
            metadatas=[{
                "id": entry['id'],
                "domain": entry['metadata']['original_domain'],
                "task_type": entry['metadata']['task_type'],
                "question": entry['input'],
                "response": entry['output'],
                "search_text": search_text,  
                "time_series": series_json
            }],
            documents=[search_text]
        )

print(f"Índice construido con {collection.count()} documentos en {CHROMA_DB_PATH}")