# 📓 Cuaderno de Bitácora del Proyecto

Archivo de registro de avances, pruebas, decisiones y notas en sucio.

---

## [2026-07-29] - Inicio del proyecto y estructura

### 🎯 Objetivos de hoy
- [x] Configurar repositorio en GitHub.
- [x] Crear estructura básica de carpetas.
- [x] Definir la estructura del README.md.

### 📝 Notas y decisiones
- Estructuré el proyecto separando `src/`, `data/` y `docs/`.
- Decidí que la documentación final (`README.md`) irá en inglés, pero mantendré este cuaderno en español para ir rápido.

### ❓ Dudas / Pendiente para mañana
- Definir qué activos exactos entran en el universo del backtest (¿S&P 500 completo o un ETF?).
- Revisar cómo limpiar los datos de precios ajustados por splits/dividendos.

---

## [29/07/2026] - Extracción de datos 

### 📝 Notas y decisiones

Descargaremos el índice S&P500 actual (con el survivorship bias que es conlleva), datos diarios (1d), horizonte temporal 2010-01-01 al 2024-12-31 (los dos últimos años lo dejamos para el backtest) y variables Open, High, Low, Close, Adj Close, Volume (aunque inicialmente solo utilicemos Adj Close). 

Los datos brutos (raw) se almacenarán exactamente con la estructura MultiIndex de yfinance (niveles Price y Ticker). Todas las fases posteriores del pipeline partirán de este dataset, realizando transformaciones específicas solo cuando sean necesarias, sin modificar nunca la fuente original.

La lista de los Tickers se obtiene de wikipedia: se crea el df sp500 con el ticker de las 503 empresas e información adicional, se modifica el formato y se extrae la lista de tickers. El sp500 se sube a data/raw para tenerlo guarado. 
...