#!/usr/bin/env python3
"""
Generador del cuaderno Jupyter específico para reprocesar P6_MULTIVARIADO en Kaggle.
Crea 'scripts/reprocesar_p6_kaggle.ipynb'.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    def code(text):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    # =========================================================================
    # CELDA 1: TÍTULO Y GUÍA DE USO
    # =========================================================================
    cells.append(md("""# 🩺 Reprocesamiento de P6_MULTIVARIADO (Kaggle GPU)
### Regeneración Sincronizada de Signos Vitales Hemodinámicos (Presión Sistólica y Diastólica)

Este cuaderno permite regenerar y evaluar el perfil **P6_MULTIVARIADO** para todos los escenarios ($E_1$ a $E_9$) utilizando las **GPUs de Kaggle (Tesla T4 o P100)**:
* **Entrada requerida:** Dataset adjunto en Kaggle con la carpeta `resultados` (que contiene las subcarpetas `E1_resultados` a `E9_resultados` con sus respectivos adaptadores `student_adapter`).
* **Salida generada:** Series sintéticas multivariadas de 2 canales, re-entrenamiento de la RNN evaluadora (LSTM), cálculo de métricas macro y desglosadas por variable, y gráficos limpios de 2 paneles (`grafica_prediccion.png`).
* **Descarga final:** Un archivo `resultados_p6_actualizados.zip` listo para descargar desde el panel de salida de Kaggle."""))

    # =========================================================================
    # CELDA 2: INSTALACIÓN DE DEPENDENCIAS
    # =========================================================================
    cells.append(md("""## 1. Instalación de Dependencias y Preparación del Entorno"""))
    cells.append(code("""# Desinstalar torchao incompatible preinstalado en Kaggle (0.10.0 vs >=0.16.0 requerido por peft)
!pip uninstall -y torchao
!pip install -q -U "transformers>=4.45.0" "peft>=0.13.0" accelerate bitsandbytes datasets matplotlib evaluate chromadb

import os
import gc
import sys
import glob
import shutil
import json
from pathlib import Path
import torch

print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")"""))

    # =========================================================================
    # CELDA 3: CLONAR REPOSITORIO
    # =========================================================================
    cells.append(md("""## 2. Clonar Repositorio del Proyecto (Rama `kaggle_experimentos`)
> Como el repositorio es privado, se requiere autenticación con un **GitHub Personal Access Token (PAT)**.
> Puedes configurarlo en Kaggle en **Add-ons -> Secrets -> GITHUB_TOKEN**, o pegarlo directamente en la variable `token_github` abajo."""))
    cells.append(code("""import os
import shutil
from pathlib import Path

REPO_DIR = Path("/kaggle/working/proyecto_tesis")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

# 1. Obtener token de GitHub (Secrets o manual)
gh_token = None
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    for secret_key in ["GITHUB_TOKEN", "GH_TOKEN", "github_token"]:
        try:
            gh_token = secrets.get_secret(secret_key)
            if gh_token:
                print(f"🔑 Token de GitHub cargado desde Kaggle Secrets ({secret_key}).")
                break
        except Exception:
            pass
except Exception:
    pass

# Si no usas Kaggle Secrets, puedes pegar tu token aquí (ej: "ghp_..."):
token_manual_gh = ""
if not gh_token and token_manual_gh:
    gh_token = token_manual_gh
    print("🔑 Usando token de GitHub configurado manualmente.")

# 2. Clonar repositorio con GIT_TERMINAL_PROMPT=0 para evitar bloqueos interactivos
if gh_token:
    print("📥 Clonando repositorio privado desde GitHub...")
    !GIT_TERMINAL_PROMPT=0 git clone -b kaggle_experimentos https://oauth2:{gh_token}@github.com/danyr59/tesis_antes.git /kaggle/working/proyecto_tesis
else:
    print("⚠️ No se detectó GITHUB_TOKEN. Intentando clonación pública...")
    !GIT_TERMINAL_PROMPT=0 git clone -b kaggle_experimentos https://github.com/danyr59/tesis_antes.git /kaggle/working/proyecto_tesis

if REPO_DIR.exists():
    %cd /kaggle/working/proyecto_tesis
    print("✅ Repositorio preparado en /kaggle/working/proyecto_tesis")
else:
    print("❌ ERROR: No se pudo clonar el repositorio.")
    print("💡 Al ser un repositorio privado, necesitas añadir tu GITHUB_TOKEN en Kaggle Secrets (Add-ons -> Secrets) o pegarlo en 'token_manual_gh'.")"""))

    # =========================================================================
    # CELDA 4: AUTENTICACIÓN HUGGING FACE
    # =========================================================================
    cells.append(md("""## 3. Autenticación de Hugging Face (para LLaMA 3.2 y Gemma 2)
> Agrega tu token en Kaggle: **Add-ons -> Secrets -> HF_TOKEN**, o pégalo directamente en la celda abajo."""))
    cells.append(code("""import os

# Intentar cargar desde Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("✅ HF_TOKEN cargado desde Kaggle Secrets.")
except Exception:
    token_manual = ""  # Ej: "hf_..."
    if token_manual:
        os.environ["HF_TOKEN"] = token_manual
        print("✅ HF_TOKEN configurado manualmente.")
    else:
        print("ℹ️ No se detectó HF_TOKEN. Si algún modelo lo requiere, configúralo en Kaggle Secrets.")

token_activo = os.environ.get("HF_TOKEN")
if token_activo:
    try:
        from huggingface_hub import login
        login(token=token_activo)
        print("✅ Sesión iniciada en huggingface_hub.")
    except Exception as e:
        print(f"⚠️ Nota login: {e}")"""))

    # =========================================================================
    # CELDA 5: PREPARAR CARPETA RESULTADOS
    # =========================================================================
    cells.append(md("""## 4. Localizar y Preparar la Carpeta `resultados`
El script busca automáticamente el dataset adjunto en `/kaggle/input` y lo copia a `/kaggle/working/resultados` para habilitar escritura."""))
    cells.append(code("""import glob
import shutil
from pathlib import Path

# Buscar la carpeta de resultados en /kaggle/input
input_candidates = glob.glob("/kaggle/input/**/E1_resultados", recursive=True) + glob.glob("/kaggle/input/**/E2_resultados", recursive=True)
resultados_working = Path("/kaggle/working/resultados")

if input_candidates:
    src_dir = Path(input_candidates[0]).parent
    print(f"📂 Dataset encontrado en: {src_dir}")
    if resultados_working.exists():
        shutil.rmtree(resultados_working)
    print("Copiando resultados a /kaggle/working/resultados (esto toma unos segundos)...")
    shutil.copytree(src_dir, resultados_working)
    print("✅ Resultados preparados en /kaggle/working/resultados")
else:
    # Si se subió como archivo ZIP
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    if zips:
        print(f"📦 Descomprimiendo archivo {zips[0]}...")
        import zipfile
        with zipfile.ZipFile(zips[0], 'r') as zip_ref:
            zip_ref.extractall("/kaggle/working/")
        print("✅ Descomprimido exitosamente.")
    else:
        print("⚠️ No se encontró la carpeta de resultados en /kaggle/input.")
        print("Por favor, asegúrate de adjuntar tu dataset 'resultados' usando el botón '+ Add Input' a la derecha.")"""))

    # =========================================================================
    # CELDA 6: EJECUTAR REPROCESAMIENTO
    # =========================================================================
    cells.append(md("""## 5. Ejecutar Reprocesamiento de P6_MULTIVARIADO
Puedes seleccionar los escenarios a procesar. Por defecto procesa los 9 escenarios (`E1,E2,E3,E4,E5,E6,E7,E8,E9`)."""))
    cells.append(code("""# Configura los escenarios que deseas reprocesar:
# Ejemplos: "E1,E2,E3,E4,E5,E6,E7,E8,E9" (todos) o "E5,E6,E7,E8,E9" (solo los restantes)
ESCENARIOS_A_PROCESAR = "E1,E2,E3,E4,E5,E6,E7,E8,E9"

!python scripts/reprocesar_p6.py \\
    --resultados_dir /kaggle/working/resultados \\
    --escenarios {ESCENARIOS_A_PROCESAR}"""))

    # =========================================================================
    # CELDA 7: VISUALIZACIÓN DE GRÁFICAS
    # =========================================================================
    cells.append(md("""## 6. Inspección de Gráficas Multivariadas Generadas"""))
    cells.append(code("""from IPython.display import Image, display

print("=" * 80)
print("📊 GRÁFICAS P6_MULTIVARIADO ACTUALIZADAS")
print("=" * 80)

for esc_id in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]:
    plot_file = Path(f"/kaggle/working/resultados/{esc_id}_resultados/P6_MULTIVARIADO/grafica_prediccion.png")
    metricas_file = Path(f"/kaggle/working/resultados/{esc_id}_resultados/P6_MULTIVARIADO/metricas.json")
    if plot_file.exists():
        r2_info = "N/A"
        if metricas_file.exists():
            with open(metricas_file) as f:
                m_data = json.load(f)
                r2_info = m_data.get("downstream", {}).get("r2", "N/A")
        print(f"\\n🎯 Escenario {esc_id} (P6_MULTIVARIADO) — R² Macro: {r2_info}")
        display(Image(filename=str(plot_file), width=850))"""))

    # =========================================================================
    # CELDA 8: EMPAQUETAR Y DESCARGAR
    # =========================================================================
    cells.append(md("""## 7. Empaquetar Resultados Actualizados para Descarga"""))
    cells.append(code("""# Comprimir la carpeta de resultados actualizada
output_zip = "/kaggle/working/resultados_p6_actualizados.zip"
if Path(output_zip).exists():
    os.remove(output_zip)

print("📦 Comprimiendo resultados actualizados...")
!cd /kaggle/working && zip -r -q resultados_p6_actualizados.zip resultados/
print("🎉 ¡Proceso completo!")
print("Puedes descargar 'resultados_p6_actualizados.zip' desde el panel derecho de Kaggle (Sección Output).")"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path("scripts/reprocesar_p6_kaggle.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Cuaderno generado exitosamente en: {out_path}")

if __name__ == "__main__":
    create_notebook()
