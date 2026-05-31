import json
import os
from pathlib import Path
from typing import List, Dict, Any
from ..utils.logger import get_logger

logger = get_logger("m4.seed_factory")

class SeedFactory:
    """
    Fábrica de datos semilla persistentes para el framework IOA.
    Genera y mantiene un dataset de sondeo (probes) de gran escala.
    """
    
    def __init__(self, seed_dir: str = "./data/seed/predictive_maintenance"):
        self.seed_dir = Path(seed_dir)
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.methodology_file = self.seed_dir / "methodology.jsonl"
        self.domain_file = self.seed_dir / "domain.jsonl"

    def get_count(self, file_path: Path) -> int:
        """Cuenta cuántos items hay en el archivo jsonl."""
        if not file_path.exists():
            return 0
        with open(file_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Extrae items de la respuesta del LLM con alta tolerancia."""
        import json
        import re
        
        response = response.strip()
        # Limpiar bloques markdown
        response = re.sub(r"```[a-zA-Z]*", "", response)
        response = response.replace("```", "").strip()
        
        try:
            data = json.loads(response)
            return data if isinstance(data, list) else [data]
        except:
            # Intento de rescate por Regex
            match = re.search(r"(\[[\s\S]*\])", response)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data if isinstance(data, list) else [data]
                except:
                    pass
            
            # Intento de rescate línea por línea (si el modelo manda objetos sueltos)
            items = []
            matches = re.findall(r"\{[\s\S]*?\}", response)
            for m in matches:
                try:
                    items.append(json.loads(m))
                except:
                    continue
            return items

    def load_seed_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Carga los datos existentes."""
        items = []
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        return items

    def append_items(self, file_path: Path, new_items: List[Dict[str, Any]]):
        """Añade nuevos items al archivo jsonl."""
        with open(file_path, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def ensure_seeds(
        self, 
        target_methodology: int, 
        target_domain: int,
        teacher_client,
        methodology_context: str,
        domain_context: str,
        data_profile: dict = None
    ):
        """Asegura que existan los items requeridos, generando los faltantes."""
        
        # Construir resumen numérico del CSV real para inyectar en prompts
        csv_summary = ""
        if data_profile:
            cols = data_profile.get("columns", [])
            stats = data_profile.get("stats", {})
            sample = data_profile.get("sample", {})
            
            csv_summary += f"COLUMNAS DEL DATASET: {', '.join(cols)}\n"
            for col in cols:
                if col in stats:
                    s = stats[col]
                    mean_val = s.get('mean', '?')
                    std_val = s.get('std', '?')
                    mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (int, float)) else str(mean_val)
                    std_str = f"{std_val:.2f}" if isinstance(std_val, (int, float)) else str(std_val)
                    csv_summary += f"  {col}: min={s.get('min','?')}, max={s.get('max','?')}, mean={mean_str}, std={std_str}\n"
            csv_summary += "\nMUESTRAS REALES DEL CSV:\n"
            if sample:
                first_col = cols[0] if cols else ""
                num_samples = len(sample.get(first_col, {}))
                for i in range(min(num_samples, 3)):
                    row_vals = [f"{col}={sample[col].get(str(i), '?')}" for col in cols]
                    csv_summary += f"  Fila {i+1}: {', '.join(row_vals)}\n"
        
        # 1. Bloque Metodológico
        current_meth = self.get_count(self.methodology_file)
        if current_meth < target_methodology:
            needed = target_methodology - current_meth
            logger.info(f"Generando {needed} items de metodología (Cerebro Maestro)...")
            new_items = self._generate_methodology_items(needed, teacher_client, methodology_context, csv_summary)
            if new_items:
                self.append_items(self.methodology_file, new_items)
            logger.info(f"Metodología actualizada: {self.get_count(self.methodology_file)} items.")

        # 2. Bloque de Dominio
        current_dom = self.get_count(self.domain_file)
        if current_dom < target_domain:
            needed = target_domain - current_dom
            logger.info(f"Generando {needed} items de dominio (Casos Clínicos)...")
            new_items = self._generate_domain_items(needed, teacher_client, domain_context, csv_summary)
            if new_items:
                self.append_items(self.domain_file, new_items)
            logger.info(f"Dominio actualizado: {self.get_count(self.domain_file)} items.")

    def _generate_methodology_items(self, count: int, client, context: str, csv_summary: str = "") -> List[Dict[str, Any]]:
        """Genera items que requieren conocimiento específico del dominio para responderse correctamente."""
        items = []
        batch_size = 5
        attempts = 0
        max_attempts = (count // batch_size + 1) * 3

        while len(items) < count and attempts < max_attempts:
            attempts += 1
            current_batch = min(batch_size, count - len(items))
            
            # Construir lista de items ya generados para evitar repeticiones
            existing_inputs = [item.get("input", "") for item in items]
            avoid_clause = ""
            if existing_inputs:
                avoid_clause = "\n\nYA GENERASTE ESTAS PREGUNTAS, NO LAS REPITAS:\n" + "\n".join(f"- {inp}" for inp in existing_inputs[-5:])

            prompt = f"""Genera {current_batch} preguntas técnicas, analíticas y MUY COMPLEJAS sobre sensores de bomba industrial. Tu objetivo es crear preguntas de nivel experto para evaluar y hacer fallar a un estudiante básico.

CONTEXTO METODOLÓGICO Y ACADÉMICO (BASA TUS PREGUNTAS EN ESTA TEORÍA):
{context}

PERFIL ESTADÍSTICO DE LOS SENSORES REALES:
{csv_summary}
{avoid_clause}

Cada pregunta debe combinar la teoría del contexto con los valores estadísticos o rangos reales del dataset. Deben ser escenarios difíciles que requieran razonamiento avanzado.

EJEMPLO DEL FORMATO ESPERADO:
[{{
  "input": "Dado que la temperatura media histórica es 75°C con desviación de 5°C, ¿qué implica una lectura sostenida de 90°C?",
  "output": "Una lectura de 90°C está 3 desviaciones estándar por encima de la media, lo que indica un posible sobrecalentamiento crítico en los rodamientos o fricción anormal.",
  "domain": "mantenimiento_predictivo",
  "module": "domain/sensor_thresholds"
}}]

RESPONDE SOLO CON EL JSON VÁLIDO:"""

            try:
                response = client.generate(prompt)
                from ..synthesis.generator import parse_llm_json_response
                new_data = parse_llm_json_response(response)
                
                if not new_data:
                    logger.warning(f"  [Intento {attempts}/{max_attempts}] Fallo de formato JSON. Reintentando...")
                else:
                    valid_items = [item for item in new_data if isinstance(item, dict) and "input" in item and "output" in item]
                    # Filtrar duplicados por input
                    unique_items = []
                    for item in valid_items:
                        if item["input"] not in existing_inputs and item["input"] not in [u["input"] for u in unique_items]:
                            unique_items.append(item)
                    
                    if not unique_items and valid_items:
                        logger.warning(f"  [Intento {attempts}/{max_attempts}] Todos los items fueron duplicados. Reintentando...")
                    
                    items.extend(unique_items[:current_batch])
                    if unique_items:
                        logger.info(f"  -> Éxito: Generados {len(unique_items)} ítems únicos (Total: {len(items)}/{count})")
                        
            except Exception as e:
                logger.warning(f"  [Intento {attempts}/{max_attempts}] Error generando metodología: {e}")
                
        if len(items) < count:
            logger.error(f"  [CRÍTICO] No se logró generar la meta de {count} items después de múltiples intentos.")
            
        return items

    def _generate_domain_items(self, count: int, client, context: str, csv_summary: str = "") -> List[Dict[str, Any]]:
        """Genera items de diagnóstico que requieren datos específicos del CSV para responderse."""
        items = []
        batch_size = 5
        attempts = 0
        max_attempts = (count // batch_size + 1) * 3
        
        # Temas para rotar y forzar diversidad
        topic_rotation = [
            "correlación entre temperatura y vibración",
            "anomalías en RPM y su relación con presión",
            "patrones de degradación temporal en los sensores",
            "umbrales críticos de fallo inminente",
            "firmas de frecuencia características de cavitación"
        ]
        
        while len(items) < count and attempts < max_attempts:
            attempts += 1
            current_batch = min(batch_size, count - len(items))
            topic = topic_rotation[(attempts - 1) % len(topic_rotation)]
            
            existing_inputs = [item.get("input", "") for item in items]
            avoid_clause = ""
            if existing_inputs:
                avoid_clause = "\n\nNO REPITAS ESTAS PREGUNTAS:\n" + "\n".join(f"- {inp}" for inp in existing_inputs[-5:])

            prompt = f"""Genera {current_batch} casos de diagnóstico extremadamente difíciles sobre: {topic}
Tu objetivo es plantear problemas de nivel ingeniero experto para hacer fallar a un estudiante básico.

CONTEXTO TEÓRICO Y DOMINIO (APLICA ESTOS CONCEPTOS A TUS CASOS):
{context}

PERFIL ESTADÍSTICO DE LOS SENSORES REALES:
{csv_summary}
{avoid_clause}

Usa valores reales del dataset (o desviaciones realistas de ellos) para construir escenarios donde múltiples variables interactúen de forma compleja según la teoría del contexto.

EJEMPLO DEL FORMATO ESPERADO:
[{{
  "input": "Lecturas: Temperatura=92°C, Vibración=3.8g, RPM=3600. ¿Cuál es el diagnóstico?",
  "output": "Fallo inminente por cavitación térmica debido a la alta temperatura y vibración anómala.",
  "domain": "mantenimiento_predictivo",
  "module": "domain/fault_signatures"
}}]

RESPONDE SOLO CON EL JSON VÁLIDO:"""
            
            try:
                response = client.generate(prompt)
                from ..synthesis.generator import parse_llm_json_response
                new_data = parse_llm_json_response(response)
                
                if not new_data:
                    logger.warning(f"  [Intento {attempts}/{max_attempts}] Fallo JSON. Reintentando...")
                else:
                    valid_items = [item for item in new_data if isinstance(item, dict) and "input" in item and "output" in item]
                    unique_items = []
                    for item in valid_items:
                        if item["input"] not in existing_inputs and item["input"] not in [u["input"] for u in unique_items]:
                            unique_items.append(item)
                    
                    if not unique_items and valid_items:
                        logger.warning(f"  [Intento {attempts}/{max_attempts}] Todos duplicados. Reintentando...")
                    
                    items.extend(unique_items[:current_batch])
                    if unique_items:
                        logger.info(f"  -> Éxito: {len(unique_items)} ítems únicos de dominio (Total: {len(items)}/{count})")
            except Exception as e:
                logger.warning(f"  [Intento {attempts}/{max_attempts}] Error generando dominio: {e}")
                
        if len(items) < count:
            logger.error(f"  [CRÍTICO] No se logró generar la meta de {count} items de dominio.")
            
        return items
