from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DB_PATH = "./../rag/maestro"
COLLECTION_NAME = "time_series_generation"

model = SentenceTransformer('all-mpnet-base-v2')
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

def consultar_maestro(consulta_usuario, top_k=3):
    """Simula la consulta del estudiante al maestro"""
    embedding = model.encode(consulta_usuario).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results

if __name__ == "__main__":
    # Simula lo que llegaría del estudiante
    consulta = "Necesito series temporales de temperatura en Mérida, Venezuela, dominio clima, frecuencia diaria, para detección de anomalías"
    resultados = consultar_maestro(consulta, top_k=200)
    
    print("El maestro ha encontrado estas series semilla:\n")
    for i, (id_, metadata, distance) in enumerate(zip(resultados['ids'][0], 
                                                      resultados['metadatas'][0],
                                                      resultados['distances'][0])):
        print(f"{i+1}. ID: {id_}")
        print(f"   Dominio: {metadata['domain']}")
        print(f"   Task type: {metadata['task_type']}")
        print(f"   question: {metadata['question']}")
        print(f"   response: {metadata['response']}")
        
        print(f"   Similitud (distancia): {distance:.4f}")
        print(f"   Fragmento: {metadata['search_text']}")
        print(f"   Time series: {metadata['time_series']}")
        print()