"""
Orquestador de auto-ingestión dual.
Separa el conocimiento del Maestro (Metodología) del conocimiento del Estudiante (Dominio).
"""

import os
from pathlib import Path
from typing import List, Optional

from .chunker import RecursiveChunker, Chunk
from .vector_store import VectorStore
from ..utils.logger import get_logger
from ..config.pipeline_config import PipelineConfig

logger = get_logger("m1.auto_ingestor")

class AutoIngestor:
    def __init__(self, config: PipelineConfig, vector_store: Optional[VectorStore] = None):
        self.config = config
        self.vs = vector_store or VectorStore(config.vector_store)
        self.chunker = RecursiveChunker()
        
    def run(self):
        """Ejecuta la ingestión de ambas carpetas."""
        logger.info(" Iniciando Auto-Ingestión Dual...")
        
        # 1. Ingestión del Cerebro del Maestro (Metodología)
        self._ingest_folder(
            folder_path="./data/methodology",
            collection_name=self.config.vector_store.collection_methodology,
            description="Cerebro del Maestro (Metodología)"
        )
        
        # 2. Ingestión del Cerebro del Estudiante (Dominio - manuales técnicos)
        self._ingest_folder(
            folder_path="./data/knowledge",
            collection_name=self.config.vector_store.collection_domain,
            description="Cerebro del Estudiante (Dominio)"
        )
        
        # 3. Ingestión de datos RAW (series temporales del CSV) al RAG del Estudiante
        self._ingest_raw_csv(
            folder_path="./data/raw",
            collection_name=self.config.vector_store.collection_domain,
        )

    def _ingest_folder(self, folder_path: str, collection_name: str, description: str):
        path = Path(folder_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Carpeta {folder_path} no existía. Creada vacía.")
            return

        logger.info(f" Procesando {description} en {folder_path}...")
        
        # Usar RecursiveChunker para procesar todo el directorio
        try:
            chunks = self.chunker.chunk_directory(path)
            if chunks:
                self.vs.add_chunks(chunks, collection_name=collection_name)
                logger.info(f"   ✓ {len(chunks)} fragmentos ingeridos en '{collection_name}'")
            else:
                logger.info(f"   ! No se encontraron documentos válidos en {folder_path}.")
        except Exception as e:
            logger.error(f"   ✗ Error en ingestión de {folder_path}: {e}")

    def _ingest_raw_csv(self, folder_path: str, collection_name: str):
        """
        Ingesta archivos CSV como fragmentos textuales en el RAG del Estudiante.
        Convierte filas numéricas en descripciones textuales que el embedding puede indexar.
        """
        import pandas as pd
        
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Carpeta {folder_path} no existe. Sin datos RAW.")
            return

        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            logger.info(f"   ! No se encontraron CSVs en {folder_path}.")
            return

        all_chunks = []
        for csv_file in csv_files:
            logger.info(f" Procesando CSV: {csv_file.name} para RAG del Estudiante...")
            try:
                df = pd.read_csv(csv_file)
                cols = list(df.columns)
                
                # Chunk 1: Resumen estadístico global
                stats_text = f"PERFIL ESTADÍSTICO DEL DATASET: {csv_file.name}\n"
                stats_text += f"Total de muestras: {len(df)}, Columnas: {', '.join(cols)}\n\n"
                for col in cols:
                    stats_text += f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    stats_text += f"media={df[col].mean():.2f}, std={df[col].std():.2f}\n"
                
                all_chunks.append(Chunk(
                    text=stats_text,
                    source=str(csv_file),
                    chunk_id=f"csv_stats_{csv_file.stem}",
                    position=0,
                    metadata={"type": "csv_statistics", "file": csv_file.name}
                ))
                
                # Chunk 2: Correlaciones entre variables
                corr = df.corr()
                corr_text = f"CORRELACIONES ENTRE VARIABLES ({csv_file.name}):\n"
                for i, col_a in enumerate(cols):
                    for col_b in cols[i+1:]:
                        corr_val = corr.loc[col_a, col_b]
                        strength = "fuerte" if abs(corr_val) > 0.7 else "moderada" if abs(corr_val) > 0.4 else "débil"
                        direction = "positiva" if corr_val > 0 else "negativa"
                        corr_text += f"  {col_a} - {col_b}: {corr_val:.3f} (correlacion {strength} {direction})\n"
                
                all_chunks.append(Chunk(
                    text=corr_text,
                    source=str(csv_file),
                    chunk_id=f"csv_correlations_{csv_file.stem}",
                    position=1,
                    metadata={"type": "csv_correlations", "file": csv_file.name}
                ))
                
                # Chunks 3+: Ventanas de datos reales (cada 25 filas)
                window_size = 25
                for start in range(0, len(df), window_size):
                    end = min(start + window_size, len(df))
                    window = df.iloc[start:end]
                    
                    window_text = f"DATOS DE SENSORES ({csv_file.name}, filas {start+1}-{end}):\n"
                    for _, row in window.iterrows():
                        row_desc = ", ".join([f"{col}={row[col]:.2f}" for col in cols])
                        window_text += f"  {row_desc}\n"
                    
                    all_chunks.append(Chunk(
                        text=window_text,
                        source=str(csv_file),
                        chunk_id=f"csv_window_{csv_file.stem}_{start}",
                        position=start // window_size + 2,
                        metadata={"type": "csv_data_window", "file": csv_file.name, "rows": f"{start+1}-{end}"}
                    ))
                
                logger.info(f"   ✓ CSV {csv_file.name}: {len(all_chunks)} fragmentos generados")
                
            except Exception as e:
                logger.error(f"   ✗ Error procesando CSV {csv_file.name}: {e}")
        
        if all_chunks:
            self.vs.add_chunks(all_chunks, collection_name=collection_name)
            logger.info(f"   ✓ {len(all_chunks)} fragmentos de datos RAW ingeridos en '{collection_name}'")
