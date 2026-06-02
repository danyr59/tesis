import os

class Settings:
    # RAG Settings
    RAG_TOP_K = 200
    RAG_COLLECTION_NAME = "time_series_generation"
    RAG_DB_PATH = "./rag/maestro"
    
    # Iteration Settings
    MAX_ITERATIONS = 3
    MIN_CONFIDENCE_THRESHOLD = 0.8
    MIN_SEED_EXAMPLES = 3
    MIN_SERIES_LENGTH = 10
    
    # Paths for Output
    SYNTHETIC_DATA_DIR = "./data/synthetic"
    
    @classmethod
    def get_teacher_model_id(cls):
        return os.getenv("TEACHER_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
        
    @classmethod
    def get_student_model_id(cls):
        return os.getenv("STUDENT_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")

    @classmethod
    def get_teacher_model_path(cls):
        return os.getenv("TEACHER_MODEL_PATH", "./models/profesor")
        
    @classmethod
    def get_student_model_path(cls):
        return os.getenv("STUDENT_MODEL_PATH", "./models/student")