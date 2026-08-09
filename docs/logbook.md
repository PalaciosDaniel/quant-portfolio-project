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

## [01/08/2026] - Feature engineering (03_feature_engineering.ipynb)

### 📝 Notas y decisiones

**Construccion de las volatilidades**

ACLARACION: Las cuatro volatilidades se encuentran justificadas en el notebook, pero el por que de utilizar 252 dias en todas las volatilidades y no 21 dias o las dos no. Se supone que la de 21 dias esta bien para hacer pruebas y tal pero no van a ser factores de nuestro modelo definitivo (igualmente como lo guardamos como funciones que admiten una parametrizacion podemos cambiarla cuando sea)

ACLARACION: Tanto la downside como la upside volatility se tiene que calular utilizando un period (que es los días seguidos que tiene que acumular para dar el dato). Hemos puesto 21 días, es decir que, cuando se acumulen 21 días de retornos positivos se da el dato de upside volatility y analogo al downside volatility (esto se hace para quedarnos solamente, a pesar de tener mas NaN porque los reotnros positivos y negativos estan intercalados, con los días de dato sólido y estable). Precisamente le hecho de poner 21 díaas en vez de 252 (que sería el caso del rolling) hace que en el analisis estadistico el count del upside y downside salga mayor que el de rolling. 

AMPLICACION: Podriamos considerar tambien las volatilidades de 21 dias pero tendriamos que simplificar de alguna manera los 8 factores para quedarnos con dos o tres como maximo. 

ACLARACION: Al final nos quedamos con tres volatilidades para alimentar el modelo (la low volaility nanai)

## [02/08/2026] - Feature engineering (03_feature_engineering.ipynb)

### 📝 Notas y decisiones

DECISIÓN: Para el Amihud se involucra la variable volumen y tenemos que ver que hacer con los nulos. Hemos decidido reemplazar los volúmenes nulos por NaN y perder el dato de ese día. Esto es mejor metodológicamente hablando que reemplazarlo por alguna media de volúmenes y tal porque el hecho de que sea nulo probablemente no sea por un fallo sino por suspensión de cotización o medio festivo. 

ACLARACION: Para el Amihud no se utiliza los retornos logarítmicos sino los simples, ya que en la formula se emplea el valor absoluto del retorno simple diario. 

DECISIÓN: Para obtener el factor de iliquidez con Amihud se deben utilizar los retornos simples. Podemos hacerlo de dos maneras: calular directamente dentro de la función los retornos simples a partir del Adj Close aprovechando que también se lo tenemos que pasar o pasarle por separado el Adj Close y los simple_returns aprovechando que los dos los tenemos en parquet. Nos hemos decidido por la segunda porque precisamente el parquet se pensó para eso, así cada vez que tenemos que darle el input a una funcion ponemos explícitamente el .parquet para traer la lista de precios, log_returns, etc. que necesitemos. 

DECISIÓN: Se ha decidido meter en la función de ``compute_amihud_illiquidity`` un mínimo de observaciones (fijadas a 15) para calcular el valor. Como es un rolling, por defecto cada vez que se encuentre un valor nulo de volumen (que nosotros lo hemos sutituido por un NaN) lo que hace es un ``skipna=True`` calulando ese valor con 20 días en vez de 21. Si esto ocurre muchos días seguidos (volúmenes nulos por suspensión varios días seguidos) podría pasar que esa media se haga entre tan pocos días que no sea estadísticamente representativo. 

ACLARACION: Todas estas features se guardan en data.processed porque son variables que salen de nuestro código, no los descargamos en ingun lado. Que esté en processed es muy diferente a la maniupacion que vamos a tener que hacer para luego darselos de comer al modelo (lo haremos mucho más tarde).

Nos hemos quedado en total con 9 features que hemos guardado en .parquet dentro de data.processed: simple returns, log returns, cumulative returns, 12-1 momentum, short term reversal, rolling volatitlity, upside volatility, downside volatility y amihud illiquidity. 

DECISION: Al final hemos incluido el sanity check y un par de tests mas de los tres tipos de retornos en el notebook del feature engineering. Mas que nada porque como tal no son los factores (de hecho todos los factores que hemos considerado dependen de estos retornos) por lo que tampoco ibamos a meter esta validacion de retornos en el notebook siguiente, que es especifico para validación de factores. 

## [02/08/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

Una vez tenemos nuestros factores calculados vamos a realizar todo ese análisis que dijimos que nos ibamos a guardar para los factores (en vez de hacerlo sobre los rendimientos), como si fuesemos nosotros mismos el modelo que intenta ver las relaciones existentes. 

ACLARACION: debemos diferenciar entre features y factores. En el parqet tenemos 9 variables pero solamente las 6 (12-1 momentum, short term reversal, rolling volatitlity, upside volatility, downside volatility y amihud illiquidity) son la comida que le daremos al modelo. 

**Diferencia entre variables base/objetivo (Base Variables / Targets) y factores de estilo (Style Factors / Alpha Drivers)**

- Los retornos son la materia prima (o el Target): simple_returns y log_returns son variables financieras base que usas para calcular otros factores (volatilidad, Sharpe, Reversal) o que usas como variable a predecir ($Y$) en tu modelo de Machine Learning (ej. predecir el retorno a $t+21$). 

- Los factores son exposiciones explicativas ($X$): un factor es una característica construida y diseñada para capturar una prima de riesgo o una anomalía de mercado (Momentum, Reversal, Volatilidad, Amihud).

El **objetivo de esta fase** es comprender las propiedades estadísticas, temporales y transversales de los factores construidos, evaluar la calidad de la información que contienen e identificar posibles redundancias antes de utilizarlos en el proceso de modelado y selección de activos. Las tres ideas importantes son: entender los factores, comprobar que aportan información útil y detectar problemas antes del modelo.

ACLARACION: Hay que entender bien cómo trabaja Downside y Upside volatility (pongamos de ejemplo la Upside). Lo que hace es filtrar los dias en que los retornos son positivos y va a trabajar con una ventana de 252 días con un mínimo de 21 días válidos. Es decir, la función mira hacia atrás en el tiempo a un periodo de 252 días y extrae los "días válidos", si estos son menos que 21, la función devuelve NaN para ese día y si son mas, calcula la volatilidad teniendo en cuenta todos los días válidos. Si por ejemplo estamos ya por el día 500, la función solo mira los últimos 252 días y hace lo mismo. La cosa es que normalmente se calula esta funcion estableciendo el min period sobre el total de la serie, no solo sobre los precios válidos, es decir, min periods=21 se asegura de que al menos 21 valores son distintos de NaN (le da igual que correspondan a retornos negativos o positivos). Es por eso que nosotros en las dos funciones hemos tranformado los retornos del signo que no estamos utilizando en valores NaN para que el min periods solo actue sobre la parte que tiene que actuar.

## [03/08/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

CUESTION: No se si deberíamos de igualar el día en el que todos los factores comienzan a dar valores distintos de NaN. es decir, ahora el momentum y la rolling comienzan en el 252 pero las otras cuatro comienzan antes. De hecho la movie es que debriamos ver si aun teniendo en cuenta las empresas que mas tarde empeizan a cotizar seguimos teniendo suficientes datos (un año despues de que la ultima empresa se ponga a cotizar) para entrenar el modelo. De hecho tampoco nos tenemos que quedar justo con el periodo en el que las 499 empresas llevan ya un año cotizando ni tampoco tenemos que cargarnos empresas porque lleven poco en el índice, simplemente lo que se puede hacer es que el modelo vaya variando el universo de empresas a medida que va pasando el tiempo (de hecho, esto es imprescindible ya que sino estariamos construyendo un modelo que no se puede actualizar). 

DECISION: Si se decide eliminar algun ticker no se toca ni el df de prices ni de ningun factor posterior calculado. Lo que se hace es ya en el notebook 05 cuando emepcemos con el modelo ahi ya vemos los tickers que eliminamos y creamos las nuevas variables prices_clean, momentum_clean, etc. que se guardaran como otro parquet en data.preprocessed/. Con las fechas limitadas para los factores lo mismo.  

## [04/08/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

ACLARACION: En el punto 4 Comportamiento Temporal y Estabilidad Cross-Sectional del notebook 04 lo que demostraremos es que los factores no solo funcionan "en promedio" durante todo el histórico, sino que mantienen su capacidad de discriminar entre activos día a día (estabilidad cross-sectional) y cómo reaccionan ante diferentes regímenes de mercado (como crisis o expansiones).

## [05/08/2026] - Factor Validation (04_factor_validation.ipynb)

### 📝 Notas y decisiones

**Cosillas sobre el análisis cross-sectional de los factores**

Lo que vamos a hacer es en lugar de mirar la serie de una sola empresa, cada día coges a las 500 empresas, miras el valor de un factor y calculas una sola cifra para todo el mercado ese día. Haces eso día a día y lo representas en una línea temporal. 

1. **Media vs. Mediana en el tiempo**: la media es el promedio habitual y la mediana es la empresa que se queda justo en el centro (el 50%). Si un factor tiene una media de $10\%$ pero una mediana de $2\%$, significa que la gran mayoría de las empresas rinden poco ($2\%$) y solo 3 o 4 gigantes inflan la media hasta el $10\%$. El peligro es que si tú construyes una estrategia de inversión pensando que la "media" del mercado es $10\%$, te vas a estrellar porque la empresa típica solo rinde el $2\%$.

Nuestro objetivo es ver si la media vive permanentemente por encima de la mediana a lo largo de los años, lo que confirmaría que la asimetría es una regla del mercado, no un accidente puntual. Esto significaría que para capturar esa media tan alta, nuestra estrategia cuantitiativa necesita identificar de forma muy fina las pocas acciones de la cola derecha (los outliers alcistas), porque si compras una acción al azar en el centro de la distribución, nos quedaríamos en la mediana (que es más baja).

2. **Dispersión: Rango Intercuartílico - IQR**: es la distancia entre el percentil 75 ($Q3$) y el percentil 25 ($Q1$), es decir, la separación entre las empresas "buenas" y las empresas "malas" ese día (si el Q75 es el deseable o es el Q25 dependerá del factor, lo que mide el IQR es la capacidad de discriminar). Si el IQR es grande (la linea es alta) hay mucha diferencia entre las empresas del top 25% y las del bottom 25% (el factor funciona porque permite diferenciar claramente unas de otras) pero si el IQR colapsa (la linea es más alta cuanta más separación haya entre quartiles y se va a cero si no hay diferencia) significa que todas las 500 empresas tienen casi el mismo valor del factor ese día por lo que el factor se ha quedado "ciego" y no te sirve para construir una cartera ese mes.

3. **Coeficiente de Variación ($\text{CV}$) y Dispersión Absoluta ($\sigma_t$ vs. $\text{IQR}_t$)**: el Coeficiente de Variación ($\text{CV} = \frac{\sigma_t}{\vert{}\mu_t\vert{}}$) busca evaluar la variabilidad relativa del factor; es decir, determinar si el aumento en la dispersión entre empresas responde a un mayor poder discriminatorio del factor o si es un mero reflejo de un entorno de mercado globalmente más volátil (como en periodos de crisis).

Nota técnica metodológica: En factores financieros cuyos rendimientos orbitan cerca de cero (como Momentum o Reversal), la media transversal ($\mu_t$) se aproxima frecuentemente a cero. Esto provoca una inestabilidad matemática en la fórmula (división por valores insignificantes) que genera picos artificiales en la serie. Por ello, se complementa la evaluación analizando de forma directa la Desviación Estándar transversal ($\sigma_t$) junto al Rango Intercuartílico ($\text{IQR}_t$), permitiendo constatar la estabilidad de la dispersión tanto en su versión paramétrica como no paramétrica sin introducir distorsiones en el denominador.

La interpretación del comportamiento conjunto de la Desviación Estándar ($\sigma_t$) y el Rango Intercuartílico ($\text{IQR}_t$) en el gráfico se resume en tres escenarios clave:

- Movimiento conjunto de $\sigma_t$ e $\text{IQR}_t$: Indica una dispersión uniforme en todo el mercado. La amplitud de la distribución se estrecha o se ensancha de manera homogénea entre todos los activos.

- Pico aislado en $\sigma_t$ mientras el $\text{IQR}_t$ se mantiene estable: Denota un efecto outlier. La dispersión paramétrica se dispara por la presencia puntual de dos o tres acciones con valores extremadamente atípicos en las colas, y no por un cambio generalizado en la mayoría del panel.

En resumen, queremos responder:  

- ¿La asimetría del factor es una propiedad constante del mercado o un espejismo temporal?

- ¿El factor sigue diferenciando entre empresas buenas y malas todos los días o se queda "muerto/ciego" en algunas épocas?

- ¿El factor se comporta de forma estable en años tranquilos y explota de forma predecible durante las crisis?

CUESTION: se ha observado una Amihud Illiquidity con tendencia bajista en los últimos 15 años debido al cambio de caracterísitcas del mercado. Deberiamos de ajustarla para poder comparar la Amihud de 2011 con la de 2014, creo que este ajuste tiene que ver con el z-score. 

**Sobre el punto 5 del notebook 04: persistencia y memoria temporal**

ACLARACION: cuando calulamos aquí el IQR mide la variación de la memoria temporal entre acciones. Queremos que sea BAJO porque indica que el factor se comporta de forma estable y predecible en todo el panel (la persistencia del factor es idéntica en casi todas las empresas del mercado, tanto para una empresa gigante como para una pequeña, la memoria temporal de su factor es prácticamente la misma). En el punto 4 estábamos midiendo la dispersión entre acciones, en este caso queremos que sea ALTO para poder clasificar (ranking).

DECISION: la correlacion que vamos a utilizar para calular lo de la correlacion cross sectional (como a medida que van pasando los lags, la serie se va pareciendo menos a la original) va a ser la de Spearman que mide el parecido en el orden en que las ordenas según el factor (rank). La de Pearson no renta mas que nada porque queremos que el modelo prediga ranking no valores concretos y adrmas porque es mucho más sensible a outliers y cambios de escala (sería muchi menos robusto). 

**Sobre el punto 6 del notebook 04: redundancia y multicolinealidad**

ACLARACION: la matriz de correlación cross-sectional mide la relación entre factores distintos en el mismo momento del tiempo, que no es lo mismo que medir la correlación en el tiempo de un mismo factor contra sí mismo (punto 5). En el punto 6 lo que queremos saber es si va a haber Redundancia Multivariada (sería responder a preguntas tipo ¿El ranking de Volatilidad Total de Apple hoy me está diciendo EXACTAMENTE LO MISMO que su ranking de Downside Volatility o su Iliquidez hoy?). 

ACLARACION: En este caso seguimos usando Spearman por las mismas razones. 

ACLARACION: Cuando hablamos del **dendrograma** lo que hacemos es convertir correlaciones de Spearman en distancias (si la correlacion es 0.90 la distancia es 0.10, muy cerquita) y vamos agrupando los factores según a que distancia estén unos de otros. Primero agrupas dos, luego esos dos con el siguiente, luego esos tres con el siguiente y al final te sale un arbol donde cuanto mas cerca esten dos ramas (o mas uniones entre ramas tengas que pasar) mas correlacion tienen.

ACLARACION: El VIF por otro lado mide cuánto puede explicarse un factor utilizando una combinación lineal de los demás factores (la matriz de correlaciones te da relaciones dos a dos, esto es todo a la vez). Lo que se hace es ajustar una regresion y se obtiene un coeficiente de determinación $R^2_j$ y el VIF se define como $VIF_j = \frac{1}{1-R^2_j}$. Si es cercano a uno entocnes ese factor aporta mucha información distinta de la que aporta el resto de factores (aporta una nueva dirección) y si es mucho mayor que uno ($VIF = 20$) entocnes ese factor no aporta apenas información nueva. 

CUESTION: Deberiamos de tener claro de todas las listas de correlaciones, matrices etc del notebook 04 cuales queremos guardar en el parquet. 

**Sobre el punto 7 del notebook 04: Preliminary Predictive Power**

ACLARACION: El Information Coefficient (IC) mide en una escala de -1 a +1 la capacidad de un factor cuantitativo para ordenar hoy los activos según la rentabilidad que tendrán en el futurO (a 21 días en nuestro caso). 

ACLARACION: conceptos importantes que se van a manejar. 

- **Mean IC**: La capacidad predictiva promedio del factor a lo largo de todo el periodo histórico.

- **Median IC**: La capacidad predictiva típica del factor, protegida de días con shocks o eventos extremos.

- **Std IC**: La variabilidad o volatilidad de la capacidad predictiva del factor día a día.

- **ICIR (Information Ratio del IC)**: La consistencia ajustada al riesgo del factor, calculada dividiendo la media del IC entre su desviación estándar.

- **% IC > 0**: El porcentaje de días en los que el factor acertó en la dirección de la rentabilidad futura.

- **t-stat**: La medida cuantitativa de cuántas desviaciones estándar se aleja el IC medio de cero para evaluar su significación.

- **p-value**: La probabilidad de que la capacidad predictiva de tu factor sea producto de la mera casualidad.

CUESTION: Probablemente en algun moemnto parte del codigo del notebook se pueda implementar como funciones que importas desde otros .py asi que si tenemos la opcion de crear ya la funcion mejor. 

## [05/08/2026] - Factor preprocessing (05_factor_preprocessing.ipynb)

### 📝 Notas y decisiones

ACLARACION: al final hemos metido en data/preprocessed los dos df que cotnienen los tres factores y los forwar returns bien limpitos en el formato correcto (diferente al original). 

DECICISON: una vez aplicado el winsorization puede ser buena contruir las dos alternativas: z score o rank percetil. Mas que nada porque hemos contruido el z score pero para modelos tipo XGBoost es mejor el rank percentil. 

## [06/08/2026] - Factor preprocessing (05_factor_preprocessing.ipynb)

### 📝 Notas y decisiones

ACLARACION: el forward return del día uno nos da la rentabilidad real que el activo generará en los siguientes 21 días laborables a partir de esa primera fecha. Literalmente mira el futuro, por eso es nuestro target. 

DECISION: Cosas que deberiamso hacer antes de empezar el notebook 06 y que desde luego por ahora no vamos a hacer. 

- Revisar que todos los notebooks tienen una estructura homogénea. 

- Comprobar que los .py asociados contienen las funciones reutilizables y que los notebooks solo orquestan el flujo.

- Actualizar el README con el estado actual del pipeline.

## [06/08/2026] - Factor modeling (06_factor_modeling.ipynb)

### 📝 Notas y decisiones

ACLARACION: Una vez tenemos los data sets subidos, a la hora de limpiarlos hacemos el dropna(). Esto no significa que nos carguemos ese día para todas las empresas sino que si una empresa tiene un NaN en un dia concreto, ese día el universo de activos se restringe para no tener en cuenta esa empresa (todas las demás, las que no tienen NaN siguen igual). Por otro lado, sí que vamos a tener que quitar los últimos 21 días de cotización (que caen en diciembre de 2024) porque no tenemos forward returns esos días por lo que no tendríamos un target al que apuntar. 

## [07/08/2026] - Factor modeling (06_factor_modeling.ipynb)

### 📝 Notas y decisiones

ACLARACION: De las mega matrices que contruimos en el notebok 05 y que cargamos en el notebook 06 nada mas empezar, que contienen toda la info de los factores y de los returns ``df_final_z`` y ``df_final_rank`` extraemos las tres columas de factores ya winsorizados y rankeados o rank percentileados: ``FEATURES_Z`` y ``FEATURES_RANK``. También extraemos el TARGET (columna de forward_returns_21d). 

Luego volvemos a juntar esos tres factores  y el target, nos cargamos todos los valores NaN y nos aseguramos de que compartan indice y así contruimos: ``df_clean_z`` y ``df_clean_rank``. Luego ya extraemos de las matrices limpitas los tres factores que serán  ``X_z`` y ``X_rank`` y los dos targets ``y_rank`` y ``y_z`` (que son idénticos). Esos X e y son los que luego spliteas en partes de validación y de train. 

ACLARACION: Para el punto 4.3 la primera tabla muestra los 7 bloques sin meter nada de la purga y el embargo (estas se meten para un fold especifico, de las 21 combinaciones que tenemos). La segunda tabla muestra una ejemlificacion digamos donde nos muestra: 

- **Total Observations**: número total de observaciones disponibles en el dataset.

- **Avg Train / Fold**: número medio de observaciones que realmente quedan para entrenar después de aplicar Purge y Embargo en un fold.

- **Avg Validation / Fold**: tamaño medio del conjunto de validación.

- **Effective Sample Reduction (%)**: porcentaje medio de observaciones que quedan excluidas en cada fold debido a las ventanas de Purga y Embargo.

EXPLICACION: En el punto 5 vamos a construir las métricas que vamos a utilizar (que guardaremos en metrics.py) para evaluar como de lejos se quedan los modelos de predecir los retornos futuros (las del 5.1 van a mirar la prediccion de los rendimientos tal cual y las del 5.2 la prediccion del ranking, que es realmente la que nos interesa). Algunas métricas importantes van a ser: 

- **RMSE (Root Mean Squared Error)**: Es la métrica principal de optimización. Lo que hace es elevar los errores al cuadrado antes de promediar, por lo que penaliza severamente los errores grandes. En gestión de carteras, equivocarte por mucho en la rentabilidad de un activo durante un evento de alta volatilidad (ej. un earnings shock o un evento macro) tiene un coste asimétrico de riesgo. El RMSE mide qué tan expuesto está el modelo a estas "pifias" graves.

- **MAE (Mean Absolute Error)**: Es una métrica de robustez y sesgo medio ya que trata todos los errores de forma lineal, sin magnificar los valores atípicos (outliers). Te da la desviación típica esperada del modelo en un día "normal" de mercado. La diferencia entre el RMSE y el MAE te dirá qué tan contaminada está la predicción por eventos extremos.

- **Rank IC (Spearman Rank Correlation)**: Es la métrica reina en la industria cuantitativa porque mide la correlación de rango de Spearman diaria entre la predicción del modelo ($\hat{y}$) y el retorno real ($y$) de todo el universo de acciones. Al ser por rangos, es totalmente inmune a los outliers del mercado (un Rank IC positivo consistente significa que el modelo sabe poner en el Top 10% las acciones que realmente van a subir más). 

- **Information Coefficient (IC - Pearson Correlation)**: Mide la correlación lineal estándar de Pearson y se utiliza para comparar con el Rank IC; si el IC difiere mucho del Rank IC, indica que la predicción está demasiado influenciada por valores extremos en la distribución de factores.

- **Information Ratio del IC ($IR_{IC}$)**: Mide la estabilidad temporal del alfa generado y se calcula como el promedio del IC diario dividido por su desviación típica:

$$\text{Information Ratio (IC)} = \frac{\mu(\text{IC}_t)}{\sigma(\text{IC}_t)}$$

En otras palabras, no queremos un modelo que tenga un IC muy alto en 2015 pero que el resto de años sea cero o negativo. Buscamos un $IR_{IC} > 0.5$ (excelente en la industria si supera $1.0$).

- **Hit Rate**: porcentaje de días en los que el IC es positivo.

- **t-stat**: evidencia estadística de que el IC medio es distinto de cero.

COMPROBACION: Cuidado con no perder de vista las funciones que vamos creando para separar en bloques la validation y train (la clase y la función split guardados en utils.py) o para evaluar los modelos (guardadas en metrics.py) porque luego vamos a necesitar llamarlas. Simplemente ahora las hemos creado para poder utilizarlas luego. 

ACLARACION: Cuando hablamos del punto 5.3 Construcción del Evaluador Modular básicamente es constuir una funcion para tener todos los examenes que quieres que tu modelo pase en una sola funcion, para que asi directamente te devuelva un data frame con todas las "notas" que ha sacado tu modelo. 

DECISION: En metrics vamos a meter todas las funciones destinadas a evaluar los modelos. Pero, el examen final que orquesta todo (el que implementamos en el 5.3) deberia ir en evaluation.py que sería el archivo donde guardamos las mega funciones que evaluan modelos (no que implementan los "examanes" sino las que corren varios a la vez y te dan las notas). 

ACLARACION: En el punto 6.1 Generic Training Function la funcion ``run_cpcv_training`` que guardamos en models/training.py básicamente se encarga evaluar un modelo que ya tiene una combinación de hiperparámetros fija. Entrena sobre los 21 folds y te da las "notas finales" del examen. Sería una manera de reusar siempre un mismo plano para los diferentes arquitectos (modelos). 

ACLARACION: Por otro lado la funcion ``optimize_hyperparameters`` (Sección 6.2) busca cuál es la combinación perfecta de hiperparámetros antes de hacer ese examen final. Pruebas decenas de combinaciones distintas sobre una muestra reducida para encontrar la ganadora.

ACLARACION: En training.py le metes las funciones para entrenar los modelos y en tuning.py las encargadas de afinar los hiperparametros. 

## [08/08/2026] - Factor modeling (06_factor_modeling.ipynb)

### 📝 Notas y decisiones

DECISION A FUTURO: Estaria bien en algun momento justificar como hemos hecho la funcion que optimiza los hiperparametros, el fundamento teorico la verdad qe ahora mismo no lo tengo nada claro: no necesitas dedicarle dos páginas a la matemática de Optuna, pero sí es importante dejar claro por qué usas TPE (Tree-structured Parzen Estimator) en lugar de una búsqueda por malla (Grid Search) o aleatoria (Random Search).

ACLARACION: Al finan en el punto 6 hemos decidido entrenar todos los modelos con los hiperparametros by default (tambien hemos metido, aparte del baseline Linear regression pura, la Ridge regression) para ver que tal lo hacen los modelos "puros". Luego ya elegimos a los mejores y ya metemos la parte de optimizacion de hiperparametros (en el punto 8). 

DECISION: Se ha manipulado la funcion run_cpcv_training ( y se ha aadidio otra antes) del training.py para adaptarla a la CPU del ordenador (ahora esa funcion ejecuta hasta 4 folds CPCV simultáneamente) y luego cada Random Forest utiliza un solo hilo. Esto se ha hecho porque Random Forest tardaba una eternidad en runear (mas de 9 minutos) pero puede ser una decision muy especifica de mi ordenador, es decir, en algun momento hay que aclarar que esto es un caso personal y a llo mejor deberiamos de generalizar. 


## [09/08/2026] - Factor modeling (06_factor_modeling.ipynb)

### 📝 Notas y decisiones

ACLARACION: Para no cargar todos los modelos otra vez, la información importante está en los .parquet. Si necesitamos meter mas codigo simplemente runeamos el codigo hasta el punto 5 incluido (ahi ya hemos creado todas las variables, importado todas las funciones, etc.) que no tarda nada (en el punto 6 ya empezamos a cargar modelos baseline).