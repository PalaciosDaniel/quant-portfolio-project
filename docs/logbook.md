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

Luego se llama a la función ``prices`` que te da un df con todos los precios de todas las empresas en el intervalo de tiempo propuesto. 

ACLARACION: al poner en la función ``auto_adjust=False`` estamos bjándonos todos los datos en crudo sin ajustar. Dentro tendríamos Open, High, Low, Close, Adj Close y Volume y para acceder a Adj Close simplemente hacemos ``prices["Adj Close"]``. Por eso ``prices`` es un df de 3773 filas (fechas) y 3018 columnas (503 empresas multiplicadas por las 6 variables). 

**Empresas con todos los valores NaN**

Tenemos cuatro empresas de las 503 que entraron después del 31/12/2024 por lo que, aunque ahora mismo estén en el índice, hasta ese momento no existirían datos. Estas son: HONA (Honeywell Aerospace), FDXF (FedEx Freight), SNDK (Sandisk) y Q (Qnity Electronics). 

La función ``validate_download`` identifica los Tickers pedidos que sí tienen columnas (o sea que no son tickers falsos ni mal nombrados), los tickers directamente mal nombrados o falsos (Missing tickers) y los pedidos que sí tienen columas pero todos sus valores son NaN (Empty tickers (NaN)) que es el caso de las cuatro empresas que hemos mencionado. 

La función ``remove_empty_tickers`` quita las empresas cuyos valores sean todos NaN en las fechas estipuladas. Obviamente si introduces esta función antes que la validación, estas cuatro empresas van a pasar de la categoría *Empty tickers (NaN)* a la categoría *Missing tickers*, pero nos asegurmos de que ya no trabajamos con ellos, quedándonos así con 499 empresas.

## [30/07/2026] - Data Quality & Cleaning
...