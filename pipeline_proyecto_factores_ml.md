# Pipeline: Modelo de factores cross-sectional con ML + construcción de cartera

## Filosofía del proyecto

Antes de entrar en fases: lo que diferencia este proyecto de un "tutorial de Kaggle con acciones" es el rigor en tres puntos que un entrevistador quant va a mirar con lupa: (1) que la validación respete el orden temporal y no haya fuga de información del futuro, (2) que el backtest sea realista (costes de transacción, turnover, sin survivorship bias no documentado), y (3) que las conclusiones sean honestas incluso cuando el modelo no funciona tan bien como esperabais. Un Sharpe espectacular sin estas precauciones es una bandera roja para cualquiera que sepa de esto; un Sharpe modesto pero bien justificado metodológicamente vale mucho más en una entrevista.

Duración estimada total: 8-10 semanas a ritmo de verano (no full-time). Está pensado para que la complejidad crezca de forma incremental, así que si el verano se complica con otras cosas, el proyecto sigue siendo presentable parándolo en cualquier fase a partir de la 6.

---

## Fase 0 — Definición de alcance y setup (3-4 días)

**Decisiones que hay que tomar antes de escribir código:**

- **Universo de activos**: empezad con algo manejable, por ejemplo los componentes actuales del S&P 500 o del Stoxx 600 (más fácil conseguir datos limpios de EEUU). Documentad desde el día 1 que estáis usando la composición *actual* del índice, no la histórica — esto introduce survivorship bias (las empresas que quebraron o fueron excluidas no aparecen). No pasa nada por tenerlo, pero hay que decirlo explícitamente en el README. Es mejor reconocer la limitación que pretender que no existe.
- **Frecuencia de rebalanceo**: mensual es el estándar académico para este tipo de modelos (datos diarios tienen demasiado ruido para factores fundamentales, datos anuales no dan suficientes observaciones). Empezad ahí.
- **Horizonte de predicción**: retorno a 1 mes vista (forward return), alineado con la frecuencia de rebalanceo.
- **Periodo histórico**: con yfinance podéis conseguir con garantías unos 10-15 años de precios diarios para casi cualquier large-cap. Apuntad a 2010-2024 o similar, dejando los últimos 1-2 años como holdout final que ninguno de los dos toca hasta el final del proyecto.

**Setup técnico:**

- Repo de GitHub desde el primer día, con estructura de carpetas clara: `data/`, `src/` (con subcarpetas `features/`, `models/`, `backtest/`), `notebooks/` (solo para exploración, el código final va en `.py`), `tests/`, `configs/`.
- Entorno reproducible: `poetry` o como mínimo un `requirements.txt` con versiones fijadas. Un quant researcher que ve un repo sin entorno reproducible asume que no habéis trabajado en equipo de verdad nunca.
- Decidid ya el control de versiones de los datos: no subáis CSVs pesados a git, usad `.gitignore` y documentad cómo regenerar los datos con un script.

**División de trabajo natural en esta fase**: tu compañero monta el esqueleto del repo y el entorno; tú documentas las decisiones de scope (universo, horizonte, frecuencia) con su justificación financiera en el README, porque ese razonamiento es justo lo que después tenéis que ser capaces de defender en una entrevista.

---

## Fase 1 — Adquisición de datos (1-1.5 semanas)

**Precios:**

`yfinance` para precios diarios ajustados (ajustados por dividendos y splits, importante para que los retornos sean correctos). Aquí ya hay una decisión sutil: el ajuste por dividendos cambia los retornos históricos, así que aseguraos de que estáis usando `Adj Close` y no `Close`.

**Datos fundamentales (para factores tipo value, quality):**

Esto es lo más difícil de conseguir gratis con calidad point-in-time (es decir, el dato tal y como se conocía en su momento, sin revisiones posteriores). Las opciones razonables para un proyecto de estudiantes:

- `yfinance` también trae algunos fundamentales pero son del momento actual, no históricos — sirven solo si simplificáis el proyecto a un único snapshot transversal (menos interesante).
- Simfin (tiene tier gratuito) ofrece históricos fundamentales con más profundidad.
- Alternativa más simple y honesta: limitad los factores fundamentales a los que se pueden derivar de precios y volumen únicamente (momentum, volatilidad, tamaño vía capitalización aproximada, liquidez), y dejad fuera factores tipo P/E o P/B si no tenéis garantías de la calidad del dato histórico. Un proyecto con 5 factores bien construidos y documentados vale más que uno con 15 factores de calidad dudosa.

**Recomendación concreta**: empezad solo con factores derivables de precio/volumen (fase 2 los detalla). Si os queda tiempo en julio/agosto, añadís factores fundamentales como extensión. Así no os bloqueáis semanas enteras peleando con APIs de datos fundamentales gratuitas, que es donde mueren muchos proyectos de este tipo.

**Estructura de almacenamiento**: un panel (panel data) con multi-índice (fecha, ticker) es la estructura natural. Pandas con `MultiIndex` o, si os apetece algo más robusto, `polars` (más rápido, y mencionarlo en el CV no hace daño porque cada vez se usa más en la industria).

**División de trabajo**: tu compañero construye el pipeline de descarga y el almacenamiento (con manejo de errores, reintentos, caché para no golpear la API cada vez); tú validas que los datos tienen sentido financiero (¿hay splits raros sin ajustar? ¿hay gaps sospechosos? ¿los retornos diarios tienen magnitudes razonables?).

---

## Fase 2 — Feature engineering: construcción de factores (1.5-2 semanas)

Esta es la fase donde tu conocimiento de finanzas pesa más. Factores típicos, derivables solo de precio/volumen:

- **Momentum**: retorno acumulado de los últimos 12 meses excluyendo el último mes (el "12-1 momentum" clásico de Jegadeesh-Titman, excluir el último mes evita el efecto de reversión a corto plazo).
- **Volatilidad**: desviación estándar de retornos diarios en una ventana móvil (por ejemplo 60 o 90 días). Los activos de baja volatilidad históricamente tienen mejor retorno ajustado a riesgo del que predicería el CAPM — el famoso "low volatility anomaly", que es un buen tema para hablar en una entrevista si os preguntan por qué incluisteis ese factor.
- **Tamaño**: log de la capitalización de mercado (precio × acciones en circulación). El "size premium" es uno de los factores de Fama-French más estudiados.
- **Liquidez**: volumen medio o algo como el Amihud illiquidity ratio.
- **Reversión a corto plazo**: retorno del último mes (efecto contrario al momentum, a más corto plazo).
- (Opcional, si conseguís fundamentales fiables) **Value**: book-to-market o earnings yield.

**Puntos técnicos importantes:**

- **Normalización cross-sectional**: en cada fecha de rebalanceo, normalizad cada factor *entre los activos de ese momento* (z-score o rank percentil), no a lo largo del tiempo. Esto es lo que hace que el modelo aprenda "qué activos son relativamente mejores en este momento" en lugar de mezclar escalas de épocas distintas.
- **Winsorización**: los factores financieros tienen colas pesadas (esto lo sabes de tus modelos GARCH/Student-t con BTC), así que recortad outliers extremos (por ejemplo al percentil 1 y 99) antes de normalizar, o el z-score se distorsiona con un solo valor extremo.
- **Lag correcto**: si usáis datos fundamentales, hay que aplicar un lag de publicación realista (los resultados del Q4 no se conocen el 1 de enero, normalmente se publican semanas o meses después). Con factores de precio/volumen este problema es menor, pero hay que tenerlo en mente para cuando añadáis fundamentales.

**División de trabajo**: tú defines y justificas financieramente cada factor (por qué debería tener poder predictivo, qué anomalía de mercado captura); tu compañero implementa el cálculo vectorizado y eficiente (evitar loops sobre miles de combinaciones fecha-ticker, usar `groupby` + `transform` en pandas o las primitivas equivalentes en polars).

---

## Fase 3 — Definición del target y construcción del dataset de entrenamiento (3-4 días)

- **Target**: forward return a 1 mes, idealmente convertido a rank percentil cross-sectional en cada fecha (esto convierte el problema en uno de *ranking relativo* más que de predicción exacta del retorno, que es mucho más difícil y menos útil — en la práctica lo que importa para construir cartera es el orden relativo, no el valor exacto).
- **Formato final**: una tabla con (fecha, ticker, factor_1, ..., factor_k, target) lista para alimentar al modelo, con un cuidado especial en que el target de la fila correspondiente a la fecha *t* use información solo hasta *t*, y el retorno objetivo sea estrictamente *t+1 mes*.
- **Missing data**: decidid una política explícita (¿eliminar filas con missing, o imputar con la mediana cross-sectional de ese factor en esa fecha?) y documentadla — es otro punto donde los entrevistadores preguntan "¿y qué hicisteis con los missing?" y "no nos dimos cuenta de que había" es la peor respuesta posible.

---

## Fase 4 — Metodología de validación temporal (clave, no os la saltéis) (3-4 días de diseño + implementación)

Este es probablemente el punto más importante de todo el proyecto desde el punto de vista de "esto demuestra que sabéis lo que hacéis".

- **Nunca k-fold aleatorio**: mezclar fechas aleatoriamente entre train y test permite que el modelo "vea" información del futuro indirectamente (por ejemplo, dos meses consecutivos del mismo activo están correlacionados, así que si uno cae en train y el contiguo en test, hay fuga).
- **Walk-forward validation**: entrenad con una ventana de datos (por ejemplo 5 años), validad/testead en el periodo inmediatamente siguiente (por ejemplo el siguiente trimestre o año), y luego desplazad la ventana hacia delante, repitiendo el proceso. Esto simula cómo se usaría el modelo en producción: solo con información pasada.
- **Embargo/purga**: dejad un pequeño hueco temporal entre el final del train y el inicio del test (por ejemplo el horizonte del target, 1 mes) para evitar fuga por el solapamiento de las ventanas de cálculo de factores y targets. Esto es una idea que viene de Marcos López de Prado (*Advances in Financial Machine Learning*) — si lo mencionáis en una entrevista, da una señal muy fuerte de que habéis leído literatura seria del campo, no solo blogs.
- **Holdout final**: el último 1-2 años de los que hablábamos en la fase 0 no se toca durante todo el desarrollo del modelo. Solo se usa una vez, al final, para el backtest definitivo. Si os encontráis "espiando" ese periodo durante el desarrollo (aunque sea para tomar decisiones de diseño), perdéis la validez de esa evaluación final — sed disciplinados con esto.

**División de trabajo**: este es un buen punto para hacerlo juntos los dos, porque es conceptualmente denso y conviene que ambos lo entendáis a fondo — es justo el tipo de cosa que os van a preguntar en una entrevista técnica y "lo hizo mi compañero" no es una respuesta aceptable aquí.

---

## Fase 5 — Modelado (1.5-2 semanas)

**Empezad simple, luego añadid complejidad:**

1. **Baseline**: regresión lineal o Elastic Net sobre los factores normalizados. Esto os da un punto de referencia interpretable (los coeficientes os dicen directamente qué factores pesan más) y es rápido de entrenar en cada ventana del walk-forward.
2. **Modelo no lineal**: LightGBM o XGBoost (LightGBM suele ser más rápido para forecasting tabular tipo panel). Estos modelos capturan interacciones entre factores que la regresión lineal no puede.
3. **Comparación honesta**: si el modelo no lineal no mejora sustancialmente al baseline lineal (es un resultado muy común en finanzas, donde la señal es débil y ruidosa), decidlo y explicad por qué puede estar pasando, en lugar de forzar una conclusión de que "el ML funciona mejor" sin evidencia.

**Métricas de evaluación del modelo en sí (antes de construir cartera):**

- **Information Coefficient (IC)**: correlación de Spearman entre las predicciones del modelo y los retornos reales observados, calculada en cada fecha de rebalanceo y promediada. Es la métrica estándar en la industria para evaluar el poder predictivo de un modelo de factores, mucho más relevante que el R² o el RMSE en este contexto.
- **Decay del IC**: ver cómo se degrada el poder predictivo si alargáis el horizonte (1 mes, 3 meses, 6 meses) — os da intuición de cuánto "dura" la señal.
- **Feature importance / SHAP**: usad SHAP values sobre el modelo de árboles para poder explicar qué factores está usando el modelo y cómo, no solo que "funciona". Esto da pie a una sección de interpretabilidad en el README que casi ningún proyecto de estudiante incluye bien.

**División de trabajo**: tu compañero lleva la implementación de los modelos y el pipeline de entrenamiento walk-forward (que es ingeniería pura: iterar ventanas, reentrenar, guardar predicciones); tú interpretas los resultados de IC y SHAP desde el punto de vista financiero (¿tiene sentido económico que el modelo se apoye en momentum más que en volatilidad en este periodo? ¿coincide con lo que se sabe de la literatura de anomalías de mercado?).

---

## Fase 6 — De predicciones a cartera (1-1.5 semanas)

Aquí entra directamente lo que ya sabes de Markowitz, pero aplicado de forma más realista:

- **Construcción simple (punto de partida)**: en cada fecha de rebalanceo, ordenad los activos por la predicción del modelo y formad carteras por deciles (long el decil superior, short el decil inferior si haces long-short; o simplemente sobreponderar el decil superior si preferís long-only, que es más realista si alguno de los dos no tiene acceso/experiencia con posiciones cortas).
- **Construcción con optimización (más interesante)**: en lugar de pesos iguales dentro de cada decil, usad las predicciones del modelo como inputs de retornos esperados en una optimización media-varianza (con la matriz de covarianza estimada de forma robusta, por ejemplo con shrinkage tipo Ledoit-Wolf en lugar de la covarianza muestral pura, que es ruidosa con pocos datos). Esto conecta directamente con la teoría de carteras de los 50 que estás estudiando, pero usando como input no los retornos históricos esperados (que es la debilidad clásica de Markowitz, "garbage in garbage out") sino las predicciones del modelo de ML — es una forma elegante y defendible de unir las dos partes del proyecto.
- **Restricciones realistas**: límite de peso máximo por activo (para no concentrar toda la cartera en 2-3 nombres), límite de turnover por rebalanceo (para controlar costes), y si hacéis long-short, neutralidad de exposición neta o sectorial si os da tiempo.

---

## Fase 7 — Motor de backtesting (1-1.5 semanas)

- **Costes de transacción**: aplicad un coste proporcional al turnover en cada rebalanceo (por ejemplo 5-10 puntos básicos por operación, valores típicos usados en literatura académica para acciones líquidas de gran capitalización). Sin esto, cualquier estrategia con rebalanceo frecuente parece mejor de lo que sería en la realidad.
- **Turnover**: calculadlo explícitamente en cada periodo (cuánto cambia la composición de la cartera) — es una métrica que cualquier entrevistador de un fondo sistemático os va a pedir si presentáis una estrategia.
- **Implementación**: evitad un backtest "vectorizado ingenuo" que no respete el orden temporal correcto entre señal, decisión y ejecución (es decir, si la señal se calcula con datos hasta el cierre del día X, la cartera no puede ejecutarse al precio de cierre del mismo día X, hay que asumir ejecución al día siguiente como mínimo). Este tipo de detalle de "look-ahead" en la ejecución es un error muy común y muy criticado cuando aparece en un backtest.

**División de trabajo**: tu compañero construye el motor de backtesting de forma genérica y reutilizable (que pueda aceptar cualquier señal de pesos y devolver la serie de retornos de cartera); tú defines qué supuestos de costes y ejecución son realistas para el tipo de activos que estáis usando.

---

## Fase 8 — Evaluación de performance (3-5 días)

Más allá del retorno total, las métricas que de verdad se miran en la industria:

- Sharpe ratio (anualizado), y si os apetece, Sortino (que solo penaliza la volatilidad negativa, más relevante para inversores reales que el Sharpe clásico).
- Maximum drawdown y tiempo de recuperación.
- Calmar ratio (retorno anualizado / max drawdown), útil para comunicar el riesgo de cola de forma simple.
- Comparación contra benchmarks honestos: no solo "buy & hold del índice", sino también contra una cartera de pesos iguales o contra el decil superior sin optimización (para aislar cuánto valor añade específicamente la parte de ML frente a simplemente seguir el ranking de un factor simple como momentum puro).
- Análisis de subperiodos: ¿la estrategia funciona de forma consistente o todo el resultado viene de 2020-2021 (un periodo muy atípico para los mercados)? Esto es clave para detectar si el resultado es robusto o es un artefacto de un periodo concreto.

---

## Fase 9 — Robustez y análisis crítico (3-5 días)

Esta fase es la que más distingue un proyecto de nivel alto de uno mediocre, y es rápida de hacer si las fases anteriores están bien construidas:

- **Sensibilidad a hiperparámetros**: ¿el resultado cambia mucho si usáis una ventana de entrenamiento de 3 años en vez de 5? ¿Si cambiáis el horizonte de predicción de 1 a 3 meses?
- **Sensibilidad al universo**: si tenéis tiempo, probad con un subconjunto distinto de activos (por ejemplo solo un sector) para ver si la señal generaliza o es específica de la composición elegida.
- **Comparación con regímenes de mercado** (aquí podéis conectar con la opción 2 de la que hablamos si os queda tiempo): ¿el modelo funciona mejor en mercados tranquilos que en mercados de alta volatilidad?

---

## Fase 10 — Documentación y storytelling para el repo (3-4 días, en paralelo con todo lo anterior idealmente)

- README con: motivación, datos usados (y limitaciones explícitas, incluido el survivorship bias), metodología de validación, resultados principales con gráficas claras (equity curve, drawdown), y una sección honesta de "qué no funcionó y por qué creemos que es así".
- Un notebook de "ejecutivo resumen" que cuente la historia completa en 10-15 minutos de lectura, separado del código de producción en `src/`.
- Mencionad explícitamente en el README las referencias metodológicas que habéis seguido (Fama-French para los factores, López de Prado para la validación temporal) — da mucha credibilidad y demuestra que no habéis improvisado la metodología.

---

## Reparto de trabajo a lo largo del verano (resumen)

Dado que tu compañero programa mejor y tú tienes más bagaje financiero, lo natural es que el peso de la ingeniería (pipelines de datos, backtesting engine, entrenamiento walk-forward) caiga más del lado suyo, y el peso de las decisiones de diseño financiero (qué factores, qué supuestos de costes, interpretación de resultados, justificación metodológica) caiga más del tuyo — pero las fases 4 (validación temporal) y 6 (de predicciones a cartera) conviene hacerlas codo a codo, porque son las que más se prestan a preguntas profundas en una entrevista y los dos debéis poder defenderlas sin titubear.

## Qué dejar fuera si el tiempo aprieta

Si llegáis a mediados de agosto y vais con retraso, lo prioritario para tener un proyecto presentable es: fases 0-8 completas con factores solo de precio/volumen (sin fundamentales) y construcción de cartera simple por deciles (sin optimización media-varianza). Las fases 9 (robustez) y la optimización avanzada de la fase 6 son las primeras que se pueden recortar sin que el proyecto pierda credibilidad, siempre que lo que quede esté bien hecho y bien documentado.
