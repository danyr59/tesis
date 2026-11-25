import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path al ejecutar como script directo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from dotenv import load_dotenv
from pipeline.student_agent.agent import StudentAgent

def main():
    # Cargar variables de entorno (Modelos, Tokens, etc.)
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Pipeline Generación Series Temporales - Maestro/Estudiante")
    parser.add_argument(
        "--query", 
        type=str, 
        help="Consulta inicial para generar series temporales",
        required=False
    )
    
    args = parser.parse_args()
    
    # Si no hay query por argumento, la pedimos por CLI
    query = args.query
    if not query:
        print("Bienvenido al Generador de Series Temporales (Agentes Maestro-Estudiante)")
        query = input("Escribe tu requerimiento (ej: necesito series temporales de temperatura): ")
        
    if not query.strip():
        print("Consulta vacía. Saliendo...")
        return
        
    # 1. Fase del Maestro (Razonamiento y Recolección)
    from pipeline.teacher_agent.agent import TeacherAgent
    import gc
    import torch

    teacher = TeacherAgent()
    structured_prompt = teacher.interpret_request(query)
    teacher_response = teacher.process_request(structured_prompt)

    # 2. Limpieza de Memoria
    print("\n[Main] Liberando memoria del Maestro...")
    
    # Liberar Embedder (SentenceTransformer) si existe
    if hasattr(teacher, 'embedder'):
        del teacher.embedder
        
    # Liberar LLM y sus componentes
    if hasattr(teacher, 'llm') and teacher.llm is not None:
        if hasattr(teacher.llm, 'model'):
            del teacher.llm.model
        if hasattr(teacher.llm, '_base_model'):
            del teacher.llm._base_model
        if hasattr(teacher.llm, 'tokenizer'):
            del teacher.llm.tokenizer
        del teacher.llm
        
    del teacher
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.empty_cache()

    # 3. Fase del Estudiante (Destilación y Generación)
    from pipeline.student_agent.agent import StudentAgent
    student = StudentAgent()
    student.run_generation(structured_prompt, teacher_response)

if __name__ == "__main__":
    main()
