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

## [31/07/2026] - Feature engineering (03_feature_engineering.ipynb)

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

## [01/07/2026] - Feature engineering (03_feature_engineering.ipynb)

### 📝 Notas y decisiones

**Construccion de las volatilidades**

ACLARACION: Las cuatro volatilidades se encuentran justificadas en el notebook, pero el por que de utilizar 252 dias en todas las volatilidades y no 21 dias o las dos no. Se supone que la de 21 dias esta bien para hacer pruebas y tal pero no van a ser factores de nuestro modelo definitivo (igualmente como lo guardamos como funciones que admiten una parametrizacion podemos cambiarla cuando sea)

ACLARACION: Tanto la downside como la upside volatility se tiene que calular utilizando un period (que es los días seguidos que tiene que acumular para dar el dato). Hemos puesto 21 días, es decir que, cuando se acumulen 21 días de retornos positivos se da el dato de upside volatility y analogo al downside volatility (esto se hace para quedarnos solamente, a pesar de tener mas NaN porque los reotnros positivos y negativos estan intercalados, con los días de dato sólido y estable). Precisamente le hecho de poner 21 díaas en vez de 252 (que sería el caso del rolling) hace que en el analisis estadistico el count del upside y downside salga mayor que el de rolling. 

AMPLICACION: Podriamos considerar tambien las volatilidades de 21 dias pero tendriamos que simplificar de alguna manera los 8 factores para quedarnos con dos o tres como maximo. 

ACLARACION: Al final nos quedamos con tres volatilidades para alimentar el modelo (la low volaility nanai)

## [02/07/2026] - Feature engineering (03_feature_engineering.ipynb)

### 📝 Notas y decisiones

DECISIÓN: Para el Amihud se involucra la variable volumen y tenemos que ver que hacer con los nulos. Hemos decidido reemplazar los volúmenes nulos por NaN y perder el dato de ese día. Esto es mejor metodológicamente hablando que reemplazarlo por alguna media de volúmenes y tal porque el hecho de que sea nulo probablemente no sea por un fallo sino por suspensión de cotización o medio festivo. 

ACLARACION: Para el Amihud no se utiliza los retornos logarítmicos sino los simples, ya que en la formula se emplea el valor absoluto del retorno simple diario. 

DECISIÓN: Para obtener el factor de iliquidez con Amihud se deben utilizar los retornos simples. Podemos hacerlo de dos maneras: calular directamente dentro de la función los retornos simples a partir del Adj Close aprovechando que también se lo tenemos que pasar o pasarle por separado el Adj Close y los simple_returns aprovechando que los dos los tenemos en parquet. Nos hemos decidido por la segunda porque precisamente el parquet se pensó para eso, así cada vez que tenemos que darle el input a una funcion ponemos explícitamente el .parquet para traer la lista de precios, log_returns, etc. que necesitemos. 

DECISIÓN: Se ha decidido meter en la función de ``compute_amihud_illiquidity`` un mínimo de observaciones (fijadas a 15) para calcular el valor. Como es un rolling, por defecto cada vez que se encuentre un valor nulo de volumen (que nosotros lo hemos sutituido por un NaN) lo que hace es un ``skipna=True`` calulando ese valor con 20 días en vez de 21. Si esto ocurre muchos días seguidos (volúmenes nulos por suspensión varios días seguidos) podría pasar que esa media se haga entre tan pocos días que no sea estadísticamente representativo. 

ACLARACION: Todas estas features se guardan en data.processed porque son variables que salen de nuestro código, no los descargamos en ingun lado. Que esté en processed es muy diferente a la maniupacion que vamos a tener que hacer para luego darselos de comer al modelo (lo haremos mucho más tarde).

Nos hemos quedado en total con 9 features que hemos guardado en .parquet dentro de data.processed: simple returns, log returns, cumulative returns, 12-1 momentum, short term reversal, rolling volatitlity, upside volatility, downside volatility y amihud illiquidity. 

DECISION: Al final hemos incluido el sanity check y un par de tests mas de los tres tipos de retornos en el notebook del feature engineering. Mas que nada porque como tal no son los factores (de hecho todos los factores que hemos considerado dependen de estos retornos) por lo que tampoco ibamos a meter esta validacion de retornos en el notebook siguiente, que es especifico para validación de factores. 

## [02/07/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

Una vez tenemos nuestros factores calculados vamos a realizar todo ese análisis que dijimos que nos ibamos a guardar para los factores (en vez de hacerlo sobre los rendimientos), como si fuesemos nosotros mismos el modelo que intenta ver las relaciones existentes. 

ACLARACION: debemos diferenciar entre features y factores. En el parqet tenemos 9 variables pero solamente las 6 (12-1 momentum, short term reversal, rolling volatitlity, upside volatility, downside volatility y amihud illiquidity) son la comida que le daremos al modelo. 

**Diferencia entre variables base/objetivo (Base Variables / Targets) y factores de estilo (Style Factors / Alpha Drivers)**

- Los retornos son la materia prima (o el Target): simple_returns y log_returns son variables financieras base que usas para calcular otros factores (volatilidad, Sharpe, Reversal) o que usas como variable a predecir ($Y$) en tu modelo de Machine Learning (ej. predecir el retorno a $t+21$). 

- Los factores son exposiciones explicativas ($X$): un factor es una característica construida y diseñada para capturar una prima de riesgo o una anomalía de mercado (Momentum, Reversal, Volatilidad, Amihud).

El **objetivo de esta fase** es comprender las propiedades estadísticas, temporales y transversales de los factores construidos, evaluar la calidad de la información que contienen e identificar posibles redundancias antes de utilizarlos en el proceso de modelado y selección de activos. Las tres ideas importantes son: entender los factores, comprobar que aportan información útil y detectar problemas antes del modelo.

ACLARACION: Hay que entender bien cómo trabaja Downside y Upside volatility (pongamos de ejemplo la Upside). Lo que hace es filtrar los dias en que los retornos son positivos y va a trabajar con una ventana de 252 días con un mínimo de 21 días válidos. Es decir, la función mira hacia atrás en el tiempo a un periodo de 252 días y extrae los "días válidos", si estos son menos que 21, la función devuelve NaN para ese día y si son mas, calcula la volatilidad teniendo en cuenta todos los días válidos. Si por ejemplo estamos ya por el día 500, la función solo mira los últimos 252 días y hace lo mismo. La cosa es que normalmente se calula esta funcion estableciendo el min period sobre el total de la serie, no solo sobre los precios válidos, es decir, min periods=21 se asegura de que al menos 21 valores son distintos de NaN (le da igual que correspondan a retornos negativos o positivos). Es por eso que nosotros en las dos funciones hemos tranformado los retornos del signo que no estamos utilizando en valores NaN para que el min periods solo actue sobre la parte que tiene que actuar.

## [03/07/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

CUESTION: No se si deberíamos de igualar el día en el que todos los factores comienzan a dar valores distintos de NaN. es decir, ahora el momentum y la rolling comienzan en el 252 pero las otras cuatro comienzan antes. De hecho la movie es que debriamos ver si aun teniendo en cuenta las empresas que mas tarde empeizan a cotizar seguimos teniendo suficientes datos (un año despues de que la ultima empresa se ponga a cotizar) para entrenar el modelo. De hecho tampoco nos tenemos que quedar justo con el periodo en el que las 499 empresas llevan ya un año cotizando ni tampoco tenemos que cargarnos empresas porque lleven poco en el índice, simplemente lo que se puede hacer es que el modelo vaya variando el universo de empresas a medida que va pasando el tiempo (de hecho, esto es imprescindible ya que sino estariamos construyendo un modelo que no se puede actualizar). 

DECISION: Si se decide eliminar algun ticker no se toca ni el df de prices ni de ningun factor posterior calculado. Lo que se hace es ya en el notebook 05 cuando emepcemos con el modelo ahi ya vemos los tickers que eliminamos y creamos las nuevas variables prices_clean, momentum_clean, etc. que se guardaran como otro parquet en data.preprocessed/. Con las fechas limitadas para los factores lo mismo.  
