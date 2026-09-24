# Preregistro — guía consciente del tiempo frente a Tweedie a lo largo del presupuesto

Escrito el 2026-09-24, **antes** de generar o evaluar ninguna run de los bloques
b6–b9. Todo lo que aquí se fija (hipótesis, métricas, comparaciones, criterios de
éxito, exclusiones) no se cambia después de ver sus resultados. Cualquier análisis
que no esté aquí se reporta como **exploratorio**.

Los resultados del barrido b1–b5 (66 runs, ya vistos) motivan estas hipótesis y
no cuentan como confirmación de ninguna de ellas.

---

## 1. Hipótesis

La intuición a contrastar: *un surrogate consciente del tiempo guía mejor a lo
largo de toda la trayectoria, converge a un MF más alto y permite recortar pasos
sin que la calidad se resienta tanto; con Tweedie hay que tener mucho más cuidado.*

Se descompone en tres hipótesis independientes. Cada una puede cumplirse o no por
separado, y el paper reporta las tres sea cual sea el resultado.

**H1 — Robustez al presupuesto.** Al bajar K, la guía noise-aware pierde menos
calidad estructural que la guía Tweedie.

**H2 — Calidad de la trayectoria de guía.** La guía noise-aware es más recta, más
coherente en el tiempo y más anticipatoria que la Tweedie, y consigue más MF por
unidad de empuje aplicado.

**H3 — Calidad final.** A igual empuje total aplicado, la guía noise-aware
alcanza un MF (FEM) mayor o igual que la Tweedie.

## 2. Evaluadores

| Código | Surrogate | Dónde se evalúa | Checkpoint |
|---|---|---|---|
| NA | PB-PUNet noise-aware | `(x_t, t)` | `engineer_solid.pt`, epoch 51 |
| HY | híbrido espectral noise-aware | `(x_t, t)` | `engineer_hybrid.pt`, epoch 35 |
| TC | PB-PUNet entrenado solo en limpio | `(x̂₀, 0)` | `clean_solid.pt`, epoch 11 |
| TS | PB-PUNet noise-aware (mismos pesos que NA) | `(x̂₀, 0)` | `engineer_solid.pt`, epoch 51 |

"Noise-aware" = {NA, HY}. "Tweedie" = {TC, TS}. La comparación principal de cada
hipótesis es **NA frente a TC** (el método del paper 1 frente al baseline DPS
estándar). NA frente a TS aísla el efecto de *dónde se evalúa* con los mismos
pesos y se reporta siempre junto a la principal. HY se reporta como segunda
realización noise-aware.

Diferencias de entrenamiento conocidas y declaradas: TC tiene 11 épocas; HY tiene
35 épocas y calendario coseno continuo. No se corrigen en b6.

## 3. Configuración fija

- Architect `architect_solid.pt` (epoch 50), v-prediction, calendario discreto.
- Sampler `discrete_vp.sample`, **eta = 1**, `clip_denoised = True`.
- Campana: `w_max = 8`, pico 0.5, anchura 0.22, `grad_clip = 5`.
- 100 muestras por run, pool de ruido inicial con semilla 20260922, lote 25
  (así el ruido ancestral queda pareado con b1–b5).
- Evaluación: Kratos, métrica `mf_resultants_area_mean`.
- Diversidad: distancia media entre pares en el latente de la VAE del paper 1.

## 4. Diseño (bloque b6)

| Sub-bloque | Contenido | Runs |
|---|---|---|
| b6 | {NA, HY, TC, TS} × γ ∈ {10, 25, 50, 100} × K ∈ {10, 20, 100} | 48 |
| b6r | {NA, HY, TC, TS} × γ = 250 × K ∈ {10, 20, 100} (referencia saturada, instrumentada) | 12 |
| b6u | sin guía × K ∈ {10, 20, 100} (gemelas: mismo ruido inicial y ancestral) | 3 |
| b6a | {NA, TC} × γ = 10 × K = 1000 (ancla: K=100 ≈ K=1000 en este régimen) | 2 |

K = 100 es el presupuesto "completo" de referencia. b6a comprueba esa elección.

## 5. Métricas

Por muestra `i`, en cada paso guiado `t` con corrección recortada `g̃_t` (la que
se suma a la velocidad), se define el desplazamiento que ese paso induce en la
estimación limpia, `d_t = −σ_t² · g̃_t` (unidades normalizadas del Architect).
`g̃` es el gradiente de la pérdida `(1 − mf)²`, así que apunta a empeorar; sumar
`σ_t·g̃` a `v` mueve `x̂₀ = α_t·x_t − σ_t·v` en `−σ_t²·g̃`.

**Primarias**

| Métrica | Definición | Hipótesis |
|---|---|---|
| `fem_mf` | MF de Kratos de la muestra final | H1, H3 |
| `straightness` | ‖Σ_t d_t‖ / Σ_t ‖d_t‖ ∈ [0, 1] | H2 |
| `push` | Σ_t ‖d_t‖ (empuje total aplicado) | H3 (eje x) |

**Secundarias**

| Métrica | Definición |
|---|---|
| `coherence` | media de cos(g̃_t, g̃_{t−1}) sobre pasos guiados consecutivos |
| `anticipation` | cos(d_t, z_i − z_i^twin) en los pasos con t ≥ 500; z^twin es la gemela sin guía |
| `efficiency` | (fem_mf_i − fem_mf_i^twin) / push_i |
| `p_bad` | fracción de muestras con fem_mf < 0.7 |
| `diversity` | distancia latente media del run |
| `fem_failures` | muestras que Kratos no resuelve |

## 6. Análisis y criterios de éxito

Todos los intervalos son **IC 95% por bootstrap pareado por muestra** (10 000
remuestreos). Las comparaciones son siempre entre runs con el mismo ruido.

**H1** — Para cada evaluador, pérdida de presupuesto `L(K) = mf(K) − mf(100)`
pareada por muestra, al mismo γ. Se cumple si, para NA frente a TC,
`L_NA(K) − L_TC(K) > 0` con IC que excluye 0 **en K = 10 y en K = 20**, para al
menos 3 de los 4 valores de γ.

**H2** — Se cumple si `straightness` de NA es mayor que la de TC, con IC que
excluye 0, **en los 12 pares (γ, K)** de b6; o en al menos 10 de 12 si en los
restantes el IC incluye 0 (ninguno a favor de TC).

**H3** — Para cada evaluador y K se interpola linealmente `fem_mf` frente a
`push` medio a través de los cuatro γ. Se comparan NA y TC en 5 valores de
`push` equiespaciados dentro del solape de sus rangos. El IC de `mf_NA − mf_TC`
se obtiene remuestreando índices de muestra (los mismos en todas las runs, para
conservar el pareado) y recalculando medias e interpolación. En cada K se
declara:

- **superior** si el límite inferior del IC es > 0 en los 5 puntos;
- **no inferior** si el límite inferior es > −0.01 en los 5 puntos;
- **inferior** en cualquier otro caso.

H3 se cumple si NA es al menos **no inferior** en K = 100 y K = 20. Si el solape
de rangos es menor que el 30% del rango de cualquiera de los dos, H3 se declara
**no contrastable** en ese K.

Un "no se cumple" no se reformula: se reporta tal cual.

## 7. Exclusiones y fallos

- Las muestras que Kratos no resuelve se excluyen de las medias de `fem_mf` y se
  reportan en `fem_failures`. En comparaciones pareadas se usa solo la
  intersección de muestras resueltas en los dos brazos.
- Análisis de sensibilidad obligatorio: repetir H1 y H3 contando cada fallo como
  `fem_mf = 0`.
- No se descarta ninguna run ni ninguna muestra por su valor.

## 8. Fases posteriores (se preregistran aquí, se ejecutan después de b6)

**Fase 3 — reentrenar HY con más peso en t bajo.** Tres variantes con 51 épocas y
calendario discreto: (i) 20% de cada lote en t = 0; (ii) peso w(t) mayor en t bajo;
(iii) ambas. **La elección se hace solo con `calibration.py`** sobre las 120
láminas de validación, nunca mirando resultados de muestreo. Pasa la variante con
error de campos ≤ 0.025 y |sesgo de MF| ≤ 0.01 para t ≤ 300, sin empeorar más de
un 10% el error de campos para t ≥ 800 respecto a HY actual. Si pasan varias, la
de menor error medio en t ≤ 300.

**Fase 4 — b7 y b8** con la variante elegida: b6 repetido para ella, y K fino
{5, 8, 12, 15} para {NA, TC, HY-nuevo} al γ con mayor `fem_mf` medio de NA en b6.

**Fase 5 — b9, confirmación con un pool de ruido nuevo** (semilla 20260923):
K ∈ {10, 20} × {NA, TC, HY-nuevo} al mismo γ, más sus gemelas. H1 y H3 se dan
por confirmadas solo si se reproducen en el signo con este pool.
