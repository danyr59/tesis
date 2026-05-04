# Manual de Metodología para Síntesis de Series Temporales
## Perfil del Maestro - Nivel: Experto Estadístico e Industrial

Este documento contiene los principios fundamentales que el Maestro debe transmitir al Estudiante para garantizar la excelencia en la generación de datos sintéticos.

### 1. Ley de Continuidad y Autocorrelación
Las series temporales en sistemas físicos (sensores, mecánica) no son secuencias de números independientes.
- **Regla:** El valor en el tiempo `t` debe ser una función de `t-1` más una variación controlada.
- **Instrucción al Aprendiz:** No permitas saltos bruscos. Si la temperatura es 80.5°C, el siguiente registro no puede ser 95°C a menos que haya un evento de falla catastrófica documentado.

### 2. Coherencia Multivariante (Física Vinculada)
En una máquina, las variables están acopladas por leyes físicas (termodinámica, fricción).
- **Regla:** Si las RPM aumentan, la temperatura y la vibración deben mostrar un incremento proporcional (correlación positiva) tras un desfase temporal lógico.
- **Instrucción al Aprendiz:** Si generas un aumento en la carga, asegúrate de que la presión y la vibración reaccionen en consecuencia. Los datos aislados sin relación entre variables son "alucinaciones físicas".

### 3. Dinámica de Ruido y Realismo
Los sistemas reales tienen imperfecciones mecánicas y eléctricas.
- **Regla:** La perfección matemática es señal de datos sintéticos pobres.
- **Instrucción al Aprendiz:** Añade un componente de "Ruido Blanco" o "Ruido Gaussiano" (pequeñas variaciones aleatorias) a cada señal. La amplitud del ruido debe ser proporcional a la estabilidad del proceso.

### 4. Estacionalidad y Ciclos Operativos
Las máquinas suelen operar en ciclos (arranque, operación estable, parada, carga variable).
- **Regla:** Identifica el ciclo operativo predominante.
- **Instrucción al Aprendiz:** Los datos deben mostrar patrones repetitivos si la operación es estable, y derivas graduales (drifts) si hay un desgaste progresivo de componentes.

### 5. Validación de Límites (Constraints)
- **Regla:** Nunca ignores los límites físicos del hardware.
- **Instrucción al Aprendiz:** Una presión no puede ser negativa, y las RPM no pueden superar el límite de diseño de la máquina sin generar una parada de emergencia inmediata.
