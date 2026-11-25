#!/usr/bin/env python3
"""
download_assets.py: Gestor automatizado de inicialización de assets (RAG y Adaptadores LoRA).

Lee automáticamente la configuración del archivo .env (TEACHER_MODEL_ID, RAG_DB_PATH, etc.)
para descargar y colocar en disco exactamente los componentes requeridos sin necesidad de
especificar parámetros en la terminal.

Uso típico:
    # 1. Configurar .env con el modelo deseado y ejecutar sin argumentos:
    python scripts/download_assets.py

    # 2. Opciones avanzadas (opcionales):
    python scripts/download_assets.py --all-adapters   # Descargar los 3 adaptadores del catálogo
    python scripts/download_assets.py --force          # Forzar re-descarga y sobreescritura
    python scripts/download_assets.py --download-base   # Descargar también checkpoints base de HF
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Resolver la raíz del repositorio
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv(REPO_ROOT / ".env")

# ── Catálogo de Modelos Docentes Recomendados (Evaluados en la Tesis) ──────
CATALOGO_MAESTROS = {
    "qwen": {
        "nombre_oficial": "Qwen2.5-7B-Instruct",
        "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
        "parametros": "7.61B",
        "contexto": "128k tokens",
        "caracteristicas": "Alta densidad simbólica, adherencia matemática estricta a esquemas JSON.",
        "kaggle_handle": "danielrangelsanch/maestro-qwen-adapter",
        "target_subdir": "qwen",
    },
    "llama": {
        "nombre_oficial": "LLaMA-3.1-8B-Instruct",
        "huggingface_id": "meta-llama/Llama-3.1-8B-Instruct",
        "parametros": "8.03B",
        "contexto": "128k tokens",
        "caracteristicas": "Alta estabilidad estructural y regularidad armónica en dinámicas periódicas.",
        "kaggle_handle": "danielrangelsanch/maestro-llama-adapter",
        "target_subdir": "llama",
    },
    "gemma": {
        "nombre_oficial": "Gemma-2-9b-it",
        "huggingface_id": "google/gemma-2-9b-it",
        "parametros": "9.24B",
        "contexto": "8k tokens (Sliding Window Attention)",
        "caracteristicas": "Estabilización por logit soft-capping y compresión de representaciones no lineales.",
        "kaggle_handle": "danielrangelsanch/maestro-gemma-adapter",
        "target_subdir": "gemma",
    },
}

RAG_DATASET_CONFIG = {
    "kaggle_handle": "danielrangelsanch/rag-maestro",
    "description": "Base de datos vectorial ChromaDB (corpus TSQA, 50 semillas empíricas reales)",
    "target_dir": REPO_ROOT / "rag" / "maestro",
    "check_file": "chroma.sqlite3",
}

def ensure_kagglehub():
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("📦 'kagglehub' no detectado. Instalándolo automáticamente...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
        return kagglehub

def detectar_familia_maestro(teacher_id: str) -> str:
    """Identifica la familia del modelo docente configurado en .env."""
    tid = teacher_id.lower()
    if "qwen" in tid:
        return "qwen"
    elif "llama" in tid:
        return "llama"
    elif "gemma" in tid:
        return "gemma"
    return "custom"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gestor automatizado de inicialización de assets (RAG y adaptadores LoRA)."
    )
    parser.add_argument(
        "--all-adapters",
        action="store_true",
        default=False,
        help="Descargar los adaptadores de los 3 Maestros (Qwen, LLaMA y Gemma) en lugar de solo el configurado en .env.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Forzar la re-descarga y sobreescritura de los assets incluso si ya existen.",
    )
    parser.add_argument(
        "--download-base",
        action="store_true",
        default=False,
        help="Descargar también los checkpoints base completos desde Hugging Face (requiere HF_TOKEN).",
    )
    return parser.parse_args()

def smart_install(src_dir: Path, target_dir: Path, ignore_checkpoints: bool = False):
    """Instala archivos usando clonación APFS Copy-on-Write en macOS (0 bytes adicionales) o copia estándar."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        try:
            import subprocess
            cmd = ["cp", "-Rc", f"{src_dir}/.", str(target_dir)]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0:
                if ignore_checkpoints:
                    for ckpt in target_dir.glob("checkpoint-*"):
                        if ckpt.is_dir():
                            shutil.rmtree(ckpt)
                        else:
                            ckpt.unlink()
                return True
        except Exception:
            pass

    for item in src_dir.iterdir():
        if ignore_checkpoints and item.name.startswith("checkpoint-"):
            continue
        dst = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    return True

def verificar_y_descargar_rag(force: bool = False):
    print("\n" + "─" * 70)
    print("1️⃣ VERIFICACIÓN Y GESTIÓN DE LA BASE VECTORIAL RAG (ChromaDB)")
    print("─" * 70)

    target_dir = RAG_DATASET_CONFIG["target_dir"]
    check_file = target_dir / RAG_DATASET_CONFIG["check_file"]

    if check_file.exists() and not force:
        print(f"✅ Base RAG ChromaDB detectada en: {target_dir}")
        print("   (Contiene las 50 semillas empíricas del corpus TSQA listas para inferencia)")
        return True

    print(f"📥 Descargando {RAG_DATASET_CONFIG['description']} desde Kaggle...")
    print(f"   Repositorio público: {RAG_DATASET_CONFIG['kaggle_handle']}")
    
    kagglehub = ensure_kagglehub()
    try:
        download_path = Path(kagglehub.dataset_download(RAG_DATASET_CONFIG["kaggle_handle"]))
        smart_install(download_path, target_dir)
        print(f"✨ Base RAG instalada con éxito en: {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Error al descargar el RAG desde Kaggle: {e}")
        return False

def verificar_y_descargar_adaptador(familia: str, force: bool = False):
    info = CATALOGO_MAESTROS[familia]
    target_dir = REPO_ROOT / "models" / "teacher_adapters" / info["target_subdir"]
    check_file = target_dir / "adapter_config.json"

    print(f"\n📦 Maestro: {info['nombre_oficial']} ({info['parametros']}, {info['contexto']})")
    print(f"   Perfil:   {info['caracteristicas']}")

    if check_file.exists() and not force:
        print(f"✅ Adaptador LoRA ya instalado en: {target_dir}")
        return True

    print(f"📥 Descargando adaptador LoRA desde Kaggle ({info['kaggle_handle']})...")
    kagglehub = ensure_kagglehub()
    try:
        download_path = Path(kagglehub.dataset_download(info["kaggle_handle"]))
        smart_install(download_path, target_dir, ignore_checkpoints=True)
        print(f"✨ Adaptador instalado satisfactoriamente en: {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Error al descargar adaptador {familia}: {e}")
        return False

def verificar_modelo_base(model_id: str, local_path_str: str = None) -> tuple[bool, str]:
    """Verifica si los pesos de un modelo base están disponibles en disco local o en la caché de Hugging Face."""
    # 1. Comprobar ruta local configurada (ej. ./models/profesor)
    if local_path_str:
        p = Path(local_path_str)
        if p.exists() and any(p.glob("*.safetensors")) or any(p.glob("*.bin")):
            return True, f"Ruta local ({local_path_str})"
            
    # 2. Comprobar caché estándar de Hugging Face
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if hf_cache.exists() and (any(hf_cache.rglob("*.safetensors")) or any(hf_cache.rglob("*.bin"))):
        return True, "Caché de Hugging Face (~/.cache/huggingface)"
        
    return False, "Pendiente de descarga desde Hugging Face"

def main():
    args = parse_args()

    teacher_id = os.getenv("TEACHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    student_id = os.getenv("STUDENT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    teacher_path = os.getenv("TEACHER_MODEL_PATH", "./models/profesor")
    student_path = os.getenv("STUDENT_MODEL_PATH", "./models/student")
    familia = detectar_familia_maestro(teacher_id)

    print("=" * 70)
    print("🚀 INICIALIZACIÓN AUTOMATIZADA DE ASSETS (RAG + ADAPTADORES LO RA)")
    print("=" * 70)
    print(f"  • Modelo Maestro (.env):   {teacher_id}")
    print(f"  • Familia Detectada:       {familia.upper()}")
    print(f"  • Modelo Estudiante (.env):{student_id}")
    print("=" * 70)

    # 1. Gestionar RAG (descarga solo si falta o si force=True)
    verificar_y_descargar_rag(force=args.force)

    # 2. Gestionar Adaptador del Maestro
    print("\n" + "─" * 70)
    print("2️⃣ VERIFICACIÓN Y GESTIÓN DE ADAPTADORES LO RA DEL MAESTRO")
    print("─" * 70)

    if args.all_adapters:
        print("🌐 Descargando adaptadores para la totalidad del catálogo oficial...")
        for fam in ["qwen", "llama", "gemma"]:
            verificar_y_descargar_adaptador(fam, force=args.force)
    elif familia in CATALOGO_MAESTROS:
        verificar_y_descargar_adaptador(familia, force=args.force)
    else:
        print(f"\n⚠️ [AVISO ACADÉMICO] El modelo docente '{teacher_id}' no pertenece")
        print("   al catálogo de 3 maestros evaluados formalmente en la tesis.")
        print("   Por tanto, no existe un adaptador preentrenado público en Kaggle.")
        print("\n💡 Para generar un adaptador LoRA para este modelo docente, ejecute:")
        print(f"   python scripts/finetune_maestro.py --model {teacher_id}")

    # 3. Descarga opcional de modelos base de Hugging Face
    if args.download_base:
        print("\n" + "─" * 70)
        print("3️⃣ DESCARGA DE MODELOS BASE DESDE HUGGING FACE")
        print("─" * 70)
        hf_token = os.getenv("HF_TOKEN")
        try:
            from huggingface_hub import snapshot_download
            print(f"📥 Descargando modelo base Maestro ({teacher_id}) en {teacher_path}...")
            snapshot_download(repo_id=teacher_id, local_dir=teacher_path, token=hf_token, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
            print("✅ Modelo Maestro base descargado.")
            print(f"📥 Descargando modelo base Estudiante ({student_id}) en {student_path}...")
            snapshot_download(repo_id=student_id, local_dir=student_path, token=hf_token, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
            print("✅ Modelo Estudiante base descargado.")
        except Exception as e:
            print(f"⚠️ Error al descargar modelos base: {e}")

    # 4. Diagnóstico Final
    teacher_base_ok, teacher_base_loc = verificar_modelo_base(teacher_id, teacher_path)
    student_base_ok, student_base_loc = verificar_modelo_base(student_id, student_path)

    print("\n" + "=" * 70)
    print("🔍 RESUMEN INTEGRAL DE ASSETS EN DISCO / CACHÉ:")
    print("=" * 70)
    rag_ok = (REPO_ROOT / "rag" / "maestro" / "chroma.sqlite3").exists()
    print(f"  • 1. Base Vectorial RAG (ChromaDB):              {'✅ LISTO' if rag_ok else '❌ FALTA'}")
    
    if familia in CATALOGO_MAESTROS:
        adapter_ok = (REPO_ROOT / "models" / "teacher_adapters" / familia / "adapter_config.json").exists()
        print(f"  • 2. Adaptador LoRA Docente ({familia.capitalize()}):              {'✅ LISTO' if adapter_ok else '❌ FALTA'}")
    else:
        print(f"  • 2. Adaptador LoRA Docente ({teacher_id}):       ⚠️ PENDIENTE DE FINE-TUNING")

    print(f"  • 3. Modelo Estudiante Base ({student_id.split('/')[-1]}): {'✅ LISTO (' + student_base_loc + ')' if student_base_ok else 'ℹ️ ' + student_base_loc}")
    print(f"  • 4. Modelo Maestro Base ({teacher_id.split('/')[-1]}):    {'✅ LISTO (' + teacher_base_loc + ')' if teacher_base_ok else 'ℹ️ ' + teacher_base_loc}")

    print("=" * 70)
    if not teacher_base_ok:
        print("💡 Nota sobre el Modelo Maestro Base:")
        print(f"   El modelo {teacher_id} (~15 GB) se descargará automáticamente desde")
        print("   Hugging Face la primera vez que se ejecute el pipeline, o puedes descargarlo")
        print("   manualmente a disco ejecutando: python scripts/setup_models.py")
        print("=" * 70)
    print("🏁 ¡Entorno verificado! Puedes ejecutar el sistema con:")
    print("   python pipeline/main.py --query 'Genera una serie temporal estacional'")
    print("=" * 70)

if __name__ == "__main__":
    main()
