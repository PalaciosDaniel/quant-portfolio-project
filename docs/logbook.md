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

## [29/07/2026] - Extracción de datos (01_data_extraction.ipynb)

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

ACLARACIÓN: no se por qué ahora también se mete como NaN en todos sus valores la empresa 'QCOM' como  "possibly delisted; no price data found  (1d 2010-01-01 -> 2024-12-31)" que es un mensaje diferente al de las otras cuatro empresas. 

ACTUALIZACION: lo he vuelto a cargar y ese quinto valor ya no me aparece como missing. 

## [30/07/2026] - Data Quality & Cleaning (02_data_cleaning.ipynb)

### 📝 Notas y decisiones

Aquí vamos a crear funciones que nos inspeccionen los datos para saber: 

- **Información general**: número de activos, número de observaciones, rango temporal, frecuencia temporal.

- **Valores perdidos**: no solo los activos completamente vacíos (ya solucionado) sino los NaN de cada empresa. 

- **Fechas duplicadas**.

- **Índice ordenado**.

- **Valores imposibles**: el Adj Close debe ser mayor que cero, el High debe ser mayor que el Low o el Volume positivo. 

- **Frecuencia**: comprobar que realmente estamos trabajando con datos diarios. 

- **Huecos largos**: decidir qué hacer con aquellas empresas que hayan empezado a cotizar muy tarde. 

**Funciones de validación de calidad de datos**

``describe_yfinance_data(df)``: resume la estructura del dataset. Identifica el número de activos, número de observaciones temporales, fechas de inicio y fin, e intenta inferir la frecuencia temporal del índice.

``summarize_date_gaps(df)``: analiza los intervalos, en días naturales, entre cada par de fechas consecutivas del índice. Devuelve cuántas veces aparece cada tipo de salto temporal, lo que permite distinguir la continuidad habitual de las sesiones bursátiles de los fines de semana, festivos o posibles fechas ausentes.

``count_nans_by_ticker(df)``: cuenta los valores ausentes (NaN) de cada empresa. El resultado agrega los valores faltantes de todos los campos disponibles, como Open, Close, High, Low y Volume.

``has_duplicate_dates(df)``: comprueba si el índice del dataset contiene fechas duplicadas. Devuelve True si existen duplicados y False en caso contrario.

``is_index_sorted(df)``: verifica que el índice temporal esté ordenado cronológicamente de forma ascendente. Devuelve True cuando las fechas están correctamente ordenadas.

``detect_impossible_values(df)``: identifica valores de mercado potencialmente inválidos. Comprueba que Open, High, Low, Close, Adj Close y Volume sean positivos, y que Low no sea mayor que High (y otras relaciones). Devuelve el número de incidencias detectadas para cada empresa.

``get_trading_periods_by_ticker(df)``: identifica la primera y la última fecha con un valor válido de Close para cada empresa. Después separa los activos en dos grupos: los que tienen datos desde la primera hasta la última fecha del dataset y los que comenzaron a cotizar más tarde, dejaron de cotizar antes o no tienen datos disponibles.

ACLARACION: El segundo archivo de jupyter (02_data_cleaning) contiene tanto las formulas del validate_data.py como las del clean_data.py. 

**Limpieza**

Se ha decidido que no se va a introducir ninguna función a este respecto porque no hace falta. Digamos que los dato s son suficientemente limpios como para no teneer que preocuparnos de limpiar, podemos pasar ya directamente al feature enginering. 

ACLARACION: podríamos comnezar ahora un análisis detallado de como se correlacionan los precios, las evouciones temporales y demás pero yo creo que va a ser mejor crearnos los factores y luego hacer ese análisis no sobre precios sino sobre factores. Realmente tiene sentido porque el modelo va a trabajar con conexiones entre factores, no con precios. 

## [31/07/2026] - Feature engineering (03_feature_engineering-ipynb)

### 📝 Notas y decisiones

**Construcción de retornos**

Se ha decidido constuir tanto los retornos simples (que es con lo que vamos a constuir la mayoría de factores) como los retornos logarítmicos (para calcular cosas de volatilidad y otras estadísticas) y los acumulativos (para reconstuir el valor del capital, etc).

ACLARACION: Cada una de las features (y cada tipo de rendimiento calulado) se guarda como un DataFrame independiente, todavía no vamos a intentar unirlos. 

**Construcción de Momentum y Reversión a corto plazo** 

En general para todas las features vamos a realizar un **sanity check** que consiste en:

- **Inspección visual**: comparación de la serie temporal de precios de una de las empresas con la evolución del factor. 

- **Conteo de nulos**: comprobar que los nulos duran lo que deben durar y que no hay huecos raros a mitad de la serie histórica. 

- **Comprobacióon de rangos**: con ``df.describe()`` verificamos que no haya valores infinitos, que la media y la mediana tienen sentido y que los valores mínimos y máximos son razonables. 


ACLARACION: No olvidarse, cuando ya tengamos todas las features calculadas, empaquetarlas y subirlas con parquet. 

AMPLIACION: Ademas de los sanity checks (sería como el nivel 1) podríamos hacer **tests unitarios** (sería el nivel 2, prueba de código automatizada que aísla y verifica que una función o componente individual funcione exactamente como se espera ante una entrada determinada).

## [01/07/2026] - Feature engineering (03_feature_engineering-ipynb)

### 📝 Notas y decisiones

**Construccion de las volatilidades**

ACLARACION: Las cuatro volatilidades se encuentran justificadas en el notebook, pero el por que de utilizar 252 dias en todas las volatilidades y no 21 dias o las dos no. Se supone que la de 21 dias esta bien para hacer pruebas y tal pero no van a ser factores de nuestro modelo definitivo (igualmente como lo guardamos como funciones que admiten una parametrizacion podemos cambiarla cuando sea)

ACLARACION: Tanto la downside como la upside volatility se tiene que calular utilizando un period (que es los días seguidos que tiene que acumular para dar el dato). Hemos puesto 21 días, es decir que, cuando se acumulen 21 días de retornos positivos se da el dato de upside volatility y analogo al downside volatility (esto se hace para quedarnos solamente, a pesar de tener mas NaN porque los reotnros positivos y negativos estan intercalados, con los días de dato sólido y estable). Precisamente le hecho de poner 21 díaas en vez de 252 (que sería el caso del rolling) hace que en el analisis estadistico el count del upside y downside salga mayor que el de rolling. 

AMPLICACION: Podriamos considerar tambien las volatilidades de 21 dias pero tendriamos que simplificar de alguna manera los 8 factores para quedarnos con dos o tres como maximo. 

ACLARACION: Al final nos quedamos con tres volatilidades para alimentar el modelo (la low volaility nanai)

## [02/07/2026] - Feature engineering (03_feature_engineering-ipynb)

### 📝 Notas y decisiones

Tiramos para el liquidity con Amihud
