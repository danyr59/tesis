import os
import traceback
from pipeline.student_agent.agent import StudentAgent
from pipeline.config.settings import Settings
from dotenv import load_dotenv

def run_batch_tests():
    load_dotenv()
    student = StudentAgent()
    
    test_cases = [
        {
            "name": "Clima Mensual (Estacional, Univariada)",
            "values": {
                "tema": "temperatura mensual promedio",
                "dominio": "climatología",
                "frecuencia": "M",
                "start_date": "2020-01-01",
                "variables": "temp_celsius",
                "tipo_serie": "univariada",
                "profile_id": "P2_ESTACIONAL",
                "num_series_configuradas": 2,
                "puntos_por_serie_configurados": 12
            },
            "seeds": [{"series": [15.2, 17.1, 22.5, 28.3, 30.1, 32.2, 29.8, 25.4, 20.1, 16.5, 14.2, 13.0]}]
        },
        {
            "name": "Ventas Financieras (Tendencia+Ruido, Univariada)",
            "values": {
                "tema": "mercado de valores, precio de accion",
                "dominio": "finanzas",
                "frecuencia": "D",
                "start_date": "2024-01-01",
                "variables": "precio_accion_usd",
                "tipo_serie": "univariada",
                "profile_id": "P4_TENDENCIA_RUIDO",
                "num_series_configuradas": 2,
                "puntos_por_serie_configurados": 20
            },
            "seeds": [{"series": [100.0, 102.5, 99.8, 105.1, 104.2]}]
        },
        {
            "name": "Tráfico Web (Complejo, Multivariado)",
            "values": {
                "tema": "trafico web de un servidor y uso de red",
                "dominio": "sistemas TI",
                "frecuencia": "H",
                "start_date": "2024-05-01 00:00:00",
                "variables": "visitas, ancho_banda_mb",
                "tipo_serie": "multivariada",
                "profile_id": "P5_COMPLEJO",
                "num_series_configuradas": 1,
                "puntos_por_serie_configurados": 15
            },
            "seeds": [{"series": {"visitas": [1000, 1200, 1100], "ancho_banda_mb": [500, 600, 550]}}]
        },
        {
            "name": "Ritmo Cardíaco (Estable, Univariada)",
            "values": {
                "tema": "ritmo cardiaco de un paciente en reposo",
                "dominio": "salud y medicina",
                "frecuencia": "S",
                "start_date": "2024-01-01 10:00:00",
                "variables": "bpm",
                "tipo_serie": "univariada",
                "profile_id": "P1_ESTABLE",
                "num_series_configuradas": 2,
                "puntos_por_serie_configurados": 15
            },
            "seeds": [{"series": [70, 72, 71, 73, 70]}]
        }
    ]
    
    success = 0
    os.makedirs("./data/synthetic/batch_tests", exist_ok=True)
    
    print("=== INICIANDO BATERÍA DE PRUEBAS DE GENERACIÓN ===")
    
    for i, test in enumerate(test_cases):
        print(f"\n--- [{i+1}/{len(test_cases)}] Probando: {test['name']} ---")
        try:
            student.num_series_target = test["values"]["num_series_configuradas"]
            student.points_per_series_target = test["values"]["puntos_por_serie_configurados"]
            
            Settings.SYNTHETIC_CSV_NAME = f"batch_test_{i+1}.csv"
            Settings.SYNTHETIC_DATA_DIR = "./data/synthetic/batch_tests"
            
            csv_path = student.generate_final_series(
                dataset=test["seeds"],
                prompt_info={"values": test["values"]},
                model_path=Settings.get_student_model_path()
            )
            print(f"✅ EXITO: CSV generado en {csv_path}")
            success += 1
        except Exception as e:
            print(f"❌ FALLO en {test['name']}: {e}")
            traceback.print_exc()
            
    print(f"\n=== RESUMEN: {success}/{len(test_cases)} pruebas exitosas ===")

if __name__ == "__main__":
    run_batch_tests()
