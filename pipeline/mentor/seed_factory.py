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
        domain_context: str
    ):
        """Asegura que existan los items requeridos, generando los faltantes."""
        
        # 1. Bloque Metodológico
        current_meth = self.get_count(self.methodology_file)
        if current_meth < target_methodology:
            needed = target_methodology - current_meth
            logger.info(f"Generando {needed} items de metodología (Cerebro Maestro)...")
            new_items = self._generate_methodology_items(needed, teacher_client, methodology_context)
            if new_items:
                self.append_items(self.methodology_file, new_items)
            logger.info(f"Metodología actualizada: {self.get_count(self.methodology_file)} items.")

        # 2. Bloque de Dominio
        current_dom = self.get_count(self.domain_file)
        if current_dom < target_domain:
            needed = target_domain - current_dom
            logger.info(f"Generando {needed} items de dominio (Casos Clínicos)...")
            new_items = self._generate_domain_items(needed, teacher_client, domain_context)
            if new_items:
                self.append_items(self.domain_file, new_items)
            logger.info(f"Dominio actualizado: {self.get_count(self.domain_file)} items.")

    def _generate_methodology_items(self, count: int, client, context: str) -> List[Dict[str, Any]]:
        """Genera items que requieren conocimiento específico del dominio para responderse correctamente."""
        items = []
        batch_size = 5 
        
        for i in range(0, count, batch_size):
            current_batch = min(batch_size, count - i)
            logger.info(f"  -> Lote de metodología {i//batch_size + 1}: {current_batch} items...")
            
            prompt = f"""Eres un Experto en Mantenimiento Predictivo Industrial.
Genera {current_batch} preguntas de diagnóstico que SOLO pueden responderse conociendo los datos técnicos específicos.

DATOS TÉCNICOS DEL SISTEMA:
{context}

INSTRUCCIONES CRÍTICAS:
1. Cada pregunta debe requerir UN VALOR NUMÉRICO ESPECÍFICO o UN UMBRAL CONCRETO de los datos.
2. Ejemplos válidos:
   - "¿Cuál es el rango normal de temperatura de operación de esta bomba?"
   - "Si la vibración supera X mm/s, ¿qué fallo indica según los datos históricos?"
   - "¿Qué correlación existe entre la presión de descarga y la vibración en condición normal?"
3. El "output" debe ser la respuesta técnica exacta basada en los datos (con valores numéricos).
4. El "module" debe ser: domain/sensor_thresholds, domain/operational_limits, domain/fault_signatures.

RESPONDE ÚNICAMENTE CON UN JSON ARRAY:
[ {{"input": "...", "output": "...", "domain": "mantenimiento_predictivo", "module": "domain/sensor_thresholds"}}, ... ]"""
            
            try:
                response = client.generate(prompt)
                from ..synthesis.generator import parse_llm_json_response
                new_data = parse_llm_json_response(response)
                items.extend(new_data[:current_batch])
            except Exception as e:
                logger.warning(f"Error generando lote de metodología: {e}")
                
        return items

    def _generate_domain_items(self, count: int, client, context: str) -> List[Dict[str, Any]]:
        """Genera items de diagnóstico que requieren datos específicos del CSV para responderse."""
        items = []
        batch_size = 5
        
        for i in range(0, count, batch_size):
            current_batch = min(batch_size, count - i)
            logger.info(f"  -> Lote de dominio {i//batch_size + 1}: {current_batch} items...")

            prompt = f"""Eres un Ingeniero Senior con acceso a los datos del sensor de esta bomba industrial.
Genera {current_batch} casos de diagnóstico basados en los datos reales del sistema.

DATOS REALES DEL SISTEMA:
{context}

INSTRUCCIONES CRÍTICAS:
1. Las preguntas deben basarse en patrones observables SOLO en estos datos específicos.
2. Ejemplos:
   - "En los datos de la bomba, cuando la temperatura supera [VALOR] °C, ¿qué patrón de vibración es indicativo de desbalanceo?"
   - "¿En qué rango de RPM aparece la firma de frecuencia característica del fallo por cavitación en este sistema?"
3. El "output" debe incluir los valores numéricos exactos del sistema (no valores genéricos).
4. El "module" debe ser: domain/fault_signatures, domain/sensor_analysis, domain/rul_estimation.

RESPONDE ÚNICAMENTE CON UN JSON ARRAY:
[ {{"input": "...", "output": "...", "domain": "mantenimiento_predictivo", "module": "domain/fault_signatures"}}, ... ]"""
            
            try:
                response = client.generate(prompt)
                from ..synthesis.generator import parse_llm_json_response
                new_data = parse_llm_json_response(response)
                items.extend(new_data[:current_batch])
            except Exception as e:
                logger.warning(f"Error generando lote de dominio: {e}")
                
        return items
