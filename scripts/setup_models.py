import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def setup_models():
    # Cargar variables de entorno
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    teacher_id = os.getenv("TEACHER_MODEL_ID")
    student_id = os.getenv("STUDENT_MODEL_ID")
    teacher_path = os.getenv("TEACHER_MODEL_PATH")
    student_path = os.getenv("STUDENT_MODEL_PATH")

    if not token or token == "tu_token_aqui":
        print("ERROR: Por favor, configura tu HF_TOKEN en el archivo .env")
        return

    print(f"--- Iniciando descarga de modelos para Tesis ---")

    # Descargar Maestro (3B)
    print(f"\n[1/2] Descargando Maestro: {teacher_id}...")
    snapshot_download(
        repo_id=teacher_id,
        local_dir=teacher_path,
        token=token,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Evitar archivos innecesarios
    )

    # Descargar Estudiante (1B)
    print(f"\n[2/2] Descargando Estudiante: {student_id}...")
    snapshot_download(
        repo_id=student_id,
        local_dir=student_path,
        token=token,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )

    print("\n--- ¡Todo listo! Modelos descargados y verificados ---")
    print(f"Profesor: {teacher_path}")
    print(f"Estudiante: {student_path}")

if __name__ == "__main__":
    setup_models()
