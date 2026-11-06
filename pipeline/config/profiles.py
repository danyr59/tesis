from typing import Dict, Any

# Definición de Perfiles Temporales extraídos de la tesis
PROFILES: Dict[str, Dict[str, Any]] = {
    "P1_ESTABLE": {
        "profile_id": "P1_ESTABLE",
        "description": "Serie temporal univariada con comportamiento estable, sin tendencia fuerte, sin estacionalidad y sin anomalías.",
        "trend_expected": "flat",
        "seasonality_expected": "none",
        "noise_expected": "low",
        "expected_anomalies": 0
    },
    "P2_ESTACIONAL": {
        "profile_id": "P2_ESTACIONAL",
        "description": "Serie temporal con estacionalidad semanal clara, sin tendencia global fuerte y sin anomalías esperadas. Ideal para ventas, tráfico, clima base.",
        "trend_expected": "flat",
        "seasonality_expected": "weekly",
        "noise_expected": "medium",
        "expected_anomalies": 0
    },
    "P3_TENDENCIA_ESTACIONAL": {
        "profile_id": "P3_TENDENCIA_ESTACIONAL",
        "description": "Serie temporal con tendencia creciente, estacionalidad semanal y presencia de anomalías aisladas.",
        "trend_expected": "increasing",
        "seasonality_expected": "weekly",
        "noise_expected": "medium",
        "expected_anomalies": 2
    },
    "P4_TENDENCIA_RUIDO": {
        "profile_id": "P4_TENDENCIA_RUIDO",
        "description": "Serie temporal con tendencia fluctuante, sin estacionalidad explícita pero con nivel de ruido medio a alto. Ideal para finanzas o sistemas caóticos.",
        "trend_expected": "decreasing",
        "seasonality_expected": "none",
        "noise_expected": "medium",
        "expected_anomalies": 0
    },
    "P5_COMPLEJO": {
        "profile_id": "P5_COMPLEJO",
        "description": "Serie temporal compleja, con tendencia, estacionalidad marcada, alto nivel de ruido y anomalías extremas.",
        "trend_expected": "increasing",
        "seasonality_expected": "weekly",
        "noise_expected": "high",
        "expected_anomalies": 3
    },
    "P6_MULTIVARIADO": {
        "profile_id": "P6_MULTIVARIADO",
        "description": "Serie temporal multivariada con dinámicas acopladas en régimen estable, variaciones fisiológicas suaves y correlación entre variables.",
        "trend_expected": "flat",
        "seasonality_expected": "none",
        "noise_expected": "low",
        "expected_anomalies": 0
    }
}

def get_profile(profile_id: str) -> Dict[str, Any]:
    if profile_id not in PROFILES:
        return PROFILES["P1_ESTABLE"] # fallback
    return PROFILES[profile_id]
