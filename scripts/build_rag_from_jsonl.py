#!/usr/bin/env python3
"""
build_rag_from_jsonl.py: Construye la colección vectorial en ChromaDB a partir del
dataset de semillas TSQA (time_series_generation.jsonl) para el Agente Maestro.

Soporta aceleración por hardware (CUDA, Apple Silicon MPS o CPU) y procesamiento
por lotes dinámico para un indexado eficiente.
"""

import sys
import os
import json
import re
import ast
import argparse
from pathlib import Path

# Resolver la raíz del repositorio de forma determinista
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import chromadb
from sentence_transformers import SentenceTransformer

# Patrón compilado para extraer listas numéricas en los textos
PATTERN_LISTA = re.compile(
    r'\[\s*(?:-?\d+(?:\.\d+)?|\'[\dA-Za-z.-]+\'|"[\dA-Za-z.-]+")'
    r'(?:\s*,\s*(?:-?\d+(?:\.\d+)?|\'[\dA-Za-z.-]+\'|"[\dA-Za-z.-]+"))*\s*\]'
)

def get_optimal_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def extract_all_time_series(text: str) -> list:
    """Extrae todas las listas numéricas del texto de forma robusta."""
    series = []
    for match in PATTERN_LISTA.finditer(text):
        try:
            lst = ast.literal_eval(match.group())
            if isinstance(lst, list):
                series.append(lst)
        except (SyntaxError, ValueError):
            pass
    return series

def clean_description(text: str) -> str:
    """Remueve las listas numéricas largas para dejar el contexto textual limpio."""
    return re.sub(r'\s+', ' ', PATTERN_LISTA.sub('', text)).strip()

def build_search_text(entry: dict) -> str:
    domain = entry.get('metadata', {}).get('original_domain', 'General')
    task = entry.get('metadata', {}).get('task_type', 'forecasting')
    question = clean_description(entry.get('input', ''))
    response = entry.get('output', '')
    return f"Dominio: {domain}. Tarea: {task}. Pregunta: {question}. Respuesta: {response}."

def main():
    parser = argparse.ArgumentParser(
        description="Indexador vectorial ChromaDB para semillas de series temporales (TSQA)."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(REPO_ROOT / "data" / "time_series_generation.jsonl"),
        help="Ruta al archivo TSQA .jsonl de entrada."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=str(REPO_ROOT / "rag" / "maestro"),
        help="Ruta de destino para la base de datos ChromaDB persistente."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="time_series_generation",
        help="Nombre de la colección en ChromaDB."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Modelo de SentenceTransformers para embeddings."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de lote para la generación de embeddings."
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        # Intentar ruta alternativa TS_Dataset.jsonl
        alt_path = REPO_ROOT / "data" / "TS_Dataset.jsonl"
        if alt_path.exists():
            jsonl_path = alt_path
        else:
            print(f"[ERROR] No se encontro el dataset en '{jsonl_path}' ni en '{alt_path}'.")
            print("Coloca el dataset TSQA en 'data/time_series_generation.jsonl' para continuar.")
            sys.exit(1)

    device = get_optimal_device()
    print(f"[INFO] Dispositivo de computo para embeddings: {device}")
    print(f"[INFO] Cargando modelo de embeddings: {args.embedding_model}...")
    model = SentenceTransformer(args.embedding_model, device=device)

    print(f"[INFO] Inicializando cliente ChromaDB en: {args.db_path}...")
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection(args.collection)
        print(f"Coleccion previa '{args.collection}' eliminada para indexacion limpia.")
    except Exception:
        pass

    collection = client.create_collection(name=args.collection)

    print(f"[INFO] Procesando dataset desde: {jsonl_path}...")
    lines_buffer = []
    entries_buffer = []
    total_indexed = 0

    def flush_batch(ids, embs, metas, docs):
        if ids:
            collection.add(
                ids=ids,
                embeddings=embs,
                metadatas=metas,
                documents=docs
            )

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            entry = json.loads(line_str)

            series_input = extract_all_time_series(entry.get('input', ''))
            series_output = extract_all_time_series(entry.get('output', ''))
            series_data = {
                "input": [{"longitud_puntos": len(x), "serie": x} for x in series_input],
                "output": [{"longitud_puntos": len(x), "serie": x} for x in series_output]
            }
            series_json = json.dumps(series_data, separators=(',', ':'))
            search_text = build_search_text(entry)

            lines_buffer.append(search_text)
            entries_buffer.append({
                "id": str(entry.get('id', total_indexed + len(entries_buffer))),
                "metadata": {
                    "id": str(entry.get('id', total_indexed + len(entries_buffer))),
                    "domain": str(entry.get('metadata', {}).get('original_domain', 'General')),
                    "task_type": str(entry.get('metadata', {}).get('task_type', 'forecasting')),
                    "question": str(entry.get('input', '')),
                    "response": str(entry.get('output', '')),
                    "time_series": series_json
                }
            })

            if len(lines_buffer) >= args.batch_size:
                embs = model.encode(lines_buffer, batch_size=args.batch_size, show_progress_bar=False)
                ids_batch = [e["id"] for e in entries_buffer]
                metas_batch = [e["metadata"] for e in entries_buffer]
                flush_batch(ids_batch, embs.tolist(), metas_batch, lines_buffer)
                total_indexed += len(ids_batch)
                print(f"  -> Indexados {total_indexed} documentos...", end="\r")
                lines_buffer.clear()
                entries_buffer.clear()

        # Resto pendiente
        if lines_buffer:
            embs = model.encode(lines_buffer, batch_size=args.batch_size, show_progress_bar=False)
            ids_batch = [e["id"] for e in entries_buffer]
            metas_batch = [e["metadata"] for e in entries_buffer]
            flush_batch(ids_batch, embs.tolist(), metas_batch, lines_buffer)
            total_indexed += len(ids_batch)
            lines_buffer.clear()
            entries_buffer.clear()

    print(f"\n[OK] Indice RAG construido: {collection.count()} documentos en '{args.db_path}' (coleccion: '{args.collection}').")

if __name__ == "__main__":
    main()
