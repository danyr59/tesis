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
        
    # Inicializar el Estudiante y arrancar el pipeline
    student = StudentAgent()
    student.run_pipeline(query)

if __name__ == "__main__":
    main()
