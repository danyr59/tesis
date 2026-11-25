import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def setup_models():
   
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    teacher_id = os.getenv("TEACHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    student_id = os.getenv("STUDENT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    teacher_path = os.getenv("TEACHER_MODEL_PATH", "./models/profesor")
    student_path = os.getenv("STUDENT_MODEL_PATH", "./models/student")

    if not token :
        print("ERROR: Por favor, configura tu HF_TOKEN en el archivo .env")
        return

    print(f"--- Iniciando descarga de modelos ---")

    # Descargar Maestro 
    print(f"\n[1/2] Descargando Maestro: {teacher_id}...")
    snapshot_download(
        repo_id=teacher_id,
        local_dir=teacher_path,
        token=token,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Evitar archivos innecesarios
    )

    # Descargar Estudiante 
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
