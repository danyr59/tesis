# 🧠 Directorio de Modelos y Adaptadores LoRA

Este directorio aloja los pesos de los modelos base y los adaptadores LoRA ligeros del Agente Maestro y el Agente Estudiante:

- `models/teacher_adapters/`: Adaptadores LoRA preentrenados del Maestro (`qwen`, `llama`, `gemma`), descargados automáticamente con `python scripts/download_assets.py`.
- `models/student_lora/`: Adaptadores temporales generados dinámicamente por el Estudiante durante el ciclo de destilación supervisada.
- `models/profesor/` y `models/student/`: Checkpoints base locales opcionales.

> [!NOTE]
> Los archivos binarios de pesos (`.safetensors`, `.bin`, `.pt`) están excluidos del seguimiento de Git para mantener el repositorio ligero. Se gestionan y descargan al vuelo mediante los scripts oficiales.
