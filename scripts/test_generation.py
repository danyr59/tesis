import json
from dotenv import load_dotenv
from pipeline.student_agent.agent import StudentAgent
from pipeline.config.settings import Settings

def test_generation():
    """
    Script de depuración rápida.
    Permite probar exclusivamente el motor de generación de series temporales
    saltándose la fase de Maestro (RAG) y la fase de Destilación (Fine-Tuning).
    """
    print("Iniciando prueba rápida de generación...")
    load_dotenv()
    
    # 1. Mock: Lo que normalmente devolvería el LLM de interpretación
    mock_prompt_info = {
        "values": {
            "tema": "ventas panadería",
            "dominio": "ventas diarias de comida",
            "frecuencia": "D",
            "start_date": "2024-01-01",
            "variables": "ventas_euros",
            "tipo_serie": "univariada",
            "profile_id": "P2_ESTACIONAL",
            "num_series_configuradas": 2,
            "puntos_por_serie_configurados": 14
        }
    }
    
    # 2. Mock: Lo que normalmente extraería el Maestro desde el RAG (Semillas)
    # Sirve para que el LLM sepa el rango de valores (ej. entre 100 y 300)
    mock_seed_dataset = [
        {
            "series": [120.5, 130.0, 125.0, 140.0, 250.0, 260.0, 115.0]
        }
    ]
    
    # 3. Inicializar el Estudiante
    student = StudentAgent()
    
    # Sobrescribimos temporalmente los valores target para que obedezca a nuestro mock
    student.num_series_target = mock_prompt_info["values"]["num_series_configuradas"]
    student.points_per_series_target = mock_prompt_info["values"]["puntos_por_serie_configurados"]
    
    print("\n--- Ejecutando generate_final_series (Sin Fine-Tuning) ---")
    # Pasamos el modelo base en lugar del fine-tuned para que sea instantáneo
    csv_path = student.generate_final_series(
        dataset=mock_seed_dataset,
        prompt_info=mock_prompt_info,
        model_path=Settings.get_student_model_path()
    )
    
    print(f"\n¡Generación completada!")
    print(f"Archivo guardado en: {csv_path}")

if __name__ == "__main__":
    test_generation()
