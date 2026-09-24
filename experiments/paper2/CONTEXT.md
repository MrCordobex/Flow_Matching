# Contexto metodológico — estudio de presupuesto de muestreo en diseño generativo de láminas

Documento de traspaso. Recoge la formulación, los hallazgos medidos y el diseño
experimental del trabajo de seguimiento al artículo publicado.

---

## 1. Qué hay y dónde

Dos carpetas, dos papeles distintos:

| Ruta | Contenido |
|---|---|
| `Escritorio/TFM/` | código y checkpoints del **artículo publicado** (paper 1) |
| `Escritorio/Flow_Matching/` | reescritura del pipeline + la suite del **paper 2** |
| `Escritorio/solid_npz.zip` | dataset, 2400 láminas sólidas |
| `TFM/comp_abaqus/` | las 40 geometrías verificadas en Abaqus del paper 1 |
| `TFM/models/` | `architect_solid.pt` (epoch 50), `engineer_solid.pt` (epoch 51) |
| `TFM/notebooks/outputs/solid_vae/` | VAE auxiliar de 2D latente para métricas de diversidad |

La carpeta `TFM/` contiene además una **extensión por tipología** (ficheros
`*_conditionated*`, subset `hole`) que el autor reserva para otro artículo y que
queda **explícitamente fuera de alcance**.

Aviso de nombres: el repositorio se llama `Flow_Matching` por motivos históricos.
El pipeline no usaba flow matching lineal; `src/tfm_shells/flow.py` es el código
original que solo sobrevive en tests.

---

## 2. Representación de datos y mecánica

Cada muestra es un mapa de elevación en rejilla cartesiana fija **64 × 64**:

- `z ∈ R^{1×H×W}`, altura sobre la proyección horizontal. Dominio del dataset:
  `z ∈ [0, 7.850908857450063]` m. Planta 10 × 10 m.
- `fz ∈ R^{1×H×W}`, carga vertical prescrita. En este dataset es **uniforme**,
  `fz = −24525 N/m²` en todos los nodos. Normalizado con `fz_min = −24525`,
  `fz_max = 0`, lo que hace que el canal de carga sea **constante a −1**: el
  Engineer no recibe información espacial de carga en estos experimentos.
- Respuesta mecánica `y ∈ R^{13×H×W}` de FEM (Abaqus en el dataset original),
  partida en tres bloques según la mecánica de láminas delgadas:
  - `y(u) = [uz]` — 1 canal, flecha fuera del plano
  - `y(m) = [ε11, ε22, ε12, n11, n22, n12]` — 6 canales, membrana
  - `y(f) = [κ11, κ22, κ12, m11, m22, m12]` — 6 canales, flexión
  - claves reales en los `.npz`: `uz`; `se11 se22 se12 sf11 sf22 sf12`;
    `sk11 sk22 sk12 sm11 sm22 sm12`, más `ds`, `dv`, `mf`

La separación en tres ramas está motivada físicamente: los campos de membrana
varían suavemente sobre la lámina, los de flexión se concentran cerca de apoyos,
bordes y transiciones de curvatura.

### Membrane Factor

Densidades de energía locales, con `ds` el área nodal:

```
w_memb = (n11·ε11 + n22·ε22 + 2·n12·ε12) · ds
w_flex = (m11·κ11 + m22·κ22 + 2·m12·κ12) · ds
mf     = w_memb / (w_memb + w_flex + ε),   recortado a [0,1]
```

**Conviven tres definiciones de MF que NO son intercambiables.** Esto es una
fuente real de error al comparar:

| Métrica | Definición |
|---|---|
| `mf_mean` | cociente de energías nativas por elemento de Kratos, promediado a nodos y luego sobre todos los nodos, apoyos incluidos |
| `mf_energy_ratio` | suma global de energías de membrana / (membrana + flexión). No es media espacial |
| `mf_resultants_area_mean` | fórmula de `Dataset_Kratos`: energías reconstruidas de N y M medios en puntos de integración, promedio por área, excluye elementos con los cuatro nodos empotrados |

**La que reproduce la convención con la que se construyó el dataset es
`mf_resultants_area_mean`.** Ver sección 6.

---

## 3. El artículo publicado (paper 1)

*Generative Design of Funicular Shells via Physics-Guided Diffusion* —
Martínez-Huertas, Ariza-García, Lourenço, Alfaro, González, Cueto, Canales.
Universidad Loyola Andalucía + Universidad de Zaragoza. 43 pp.

### Tesis central: el desajuste de Jensen a nivel de operador

La práctica estándar en difusión guiada por física evalúa el operador físico
sobre una estimación limpia obtenida por la fórmula de Tweedie,
`x̂₀(x_t,t) ≈ E[x₀|x_t,t]`. Para mecánica de láminas eso es delicado porque el
operador de respuesta `R` es **no lineal** en la geometría: la flexión depende
de la curvatura y el MF sale de combinaciones no lineales de deformaciones,
fuerzas, curvaturas y momentos. En general:

```
R(E[x₀|x_t,t]) ≠ E[R(x₀)|x_t,t]
```

La propuesta del paper: en vez de acercar la evaluación física a estados
limpios (destilación, unrolling), **hacer el surrogate consciente del ruido**,
entrenándolo directamente sobre pares `(x_t, t, R(x₀))` del mismo proceso
forward que entrena el prior. Bajo el argumento MSE supervisado estándar, el
predictor óptimo aproxima `E[R(x₀)|x_t,t]`.

### Arquitectura de dos ramas

**Architect** — prior geométrico incondicional.
- `UNet2DModel` de Diffusers, **120,2 M** parámetros, 1 canal entrada y salida
- `block_out_channels = [128,128,256,256,512,512]`, `layers_per_block = 2`
- 3 × `DownBlock2D` + 3 × `AttnDownBlock2D`, simétrico en subida
- calendario `squaredcos_cap_v2`, `prediction_type = v_prediction`, **T = 1000**
- embedding sinusoidal de timestep, entero en 0..999
- **no ve `fz`**: aprende `p_θ(x₀)` incondicional
- checkpoint disponible: epoch 50

**Engineer** — surrogate mecánico condicionado al tiempo.
- **PB-PUNet**: tres `UNet2DModel` en paralelo, pesos separados, misma topología
  que el Architect, entrada de 2 canales `concat(x_t, f̃z)`
- salidas 1 / 6 / 6 canales → 13. Total **360,8 M** parámetros
- entrenado sobre los **mismos estados ruidosos y el mismo calendario** que el
  Architect, condicionado a `t` por el mismo embedding
- pérdida: MSE supervisada contra los campos FEM + residuo de trabajo virtual
  `L_phys = (ΔP)²` con `ΔP = Σ(w_memb + w_flex − w_ext)`, ponderado
  `ω(t) = (1 − t/T)^p`, `p = 2`, más fuerte cerca de la geometría limpia
- checkpoint disponible: epoch 51

Los dos se entrenan **independientemente** y se acoplan **solo en inferencia**.

### Parametrización v

Objetivo `v_t = √ᾱ_t·ε − √(1−ᾱ_t)·x₀`. En forma angular, con
`cos φ_t = √ᾱ_t` y `sin φ_t = √(1−ᾱ_t)`:

```
x_t = cos φ_t · x₀ + sin φ_t · ε
v_t = dx_t/dφ_t = −sin φ_t · x₀ + cos φ_t · ε
```

Recuperación: `x̂₀ = √ᾱ_t·x_t − √(1−ᾱ_t)·v̂` y `ε̂ = √(1−ᾱ_t)·x_t + √ᾱ_t·v̂`.

La justificación del paper es la propagación de error a la geometría implícita:

```
E‖x₀ − x̂₀‖²      = (1−ᾱ_t) · E‖v − v_θ‖²        (acotado en [0,1])
E‖x₀ − x̂₀^ε‖²    = ((1−ᾱ_t)/ᾱ_t) · E‖ε − ε_θ‖²  (diverge cuando ᾱ_t → 0)
```

Con `ε`-predicción, errores moderados a ruido alto se amplifican sin cota en la
geometría implícita. Eso importa porque la guía física actúa sobre estados
intermedios.

### Guiado por Membrane Factor con campana

Objetivo estructural, invariante al tamaño de lote:

```
J_t(x_t) = (1/B) Σ_b (1 − m̂_t^(b))²      con m̂ = media espacial del mapa mf
g_t      = ∇_{x_t} J_t
```

El gradiente se calcula con autograd retropropagando por el Engineer congelado
y por el mapeo afín diferenciable entre los dominios normalizados del Architect
y del Engineer.

Programación en campana sobre el progreso inverso `r_k = k/(K−1)`, `k = 0..K−1`
desde ruido alto a ruido bajo:

```
s_k = w_max · exp(−(r_k − ρ)² / (2σ²))
g̃_t = clip(γ · s_k · g_t, −c, +c)
v_guiado = v_θ(x_t,t) + √(1−ᾱ_t) · g̃_t
```

El prefactor `√(1−ᾱ_t)` expresa la corrección en la misma parametrización `v`.
El signo produce descenso del objetivo en la estimación limpia:

```
Δx̂₀ = −(1 − ᾱ_t) · g̃_t
```

Configuración publicada: `guidance_scale γ = 250`, `guide_w_max = 8` (producto
efectivo **2000**), `grad_clip c = 5`, `ρ = 0.5`, `σ = 0.22`,
`num_inference_steps = 1000`, **`clip_sample = True`** con rango 1.0, sampler
**DDPM ancestral**.

### Resultados publicados

| Estrategia | mf | σ_mf | P(mf>0.90) | Área hull | d_lat |
|---|---|---|---|---|---|
| Sin guiado | 0.672 | 0.260 | 0.36 | 15.81 | 1.657 |
| Bell 0.8 | 0.948 | 0.093 | 0.83 | 13.33 | 1.460 |
| **Bell 0.5** | **0.960** | 0.050 | 0.89 | 9.18 | 1.276 |
| Constante | 0.951 | 0.046 | 0.91 | 6.90 | 0.973 |
| Lineal | 0.957 | 0.056 | 0.85 | 11.05 | 1.377 |

Barrido de `γ·w_max`: satura en ~0.96 desde 50; el score equilibrado
calidad/diversidad pica en **γ·w_max = 10** (96% del MF asintótico conservando
96% de la cobertura latente).

Diagnóstico de proveedores de gradiente (sección 5.1), MSE del tensor físico:

| t | noise-aware | Tweedie/DPS | clean-on-x_t |
|---|---|---|---|
| 0 | 2.47e6 | 1.84e6 | 1.84e6 |
| 600 | 8.40e6 | 9.31e6 | 5.23e7 |
| 999 | 2.18e7 | 2.62e7 | 7.29e7 |

Factor de crecimiento de t=0 a t=999: **8.8×** el noise-aware, **39.7×** el
clean-on-x_t.

**Verificación independiente en Abaqus**, 40 láminas:

| γ·w_max | n | Predicho | Abaqus | Sesgo |
|---|---|---|---|---|
| 10 | 20 | 0.978 | 0.913 | +0.065 |
| 250 | 20 | 0.981 | 0.921 | +0.060 |
| Todas | 40 | 0.979 | **0.917** | **+0.062** |

Correlación muestra a muestra 0.852. **El Engineer sobreestima el MF en ~+0.06.**

---

## 4. Hallazgo 1 — flow matching sobre esta trayectoria es el mismo modelo

Se exploró reformular el Architect como flow matching. Sobre el interpolante
coseno, la velocidad de flow matching es:

```
u(x_t,t) = dx_t/dt = (dφ/dt)·(dx_t/dφ) = k · v,    k = π / (2(1+s))
```

**`dφ/dt` es constante**, luego `u = k·v` con `k = 1.5583` para `s = 0.008`.
Es la misma función reescalada por un escalar. Consecuencias:

1. Reparametrizar a flow matching **no cambia el campo aprendido**. No se puede
   reclamar como contribución: un revisor lo verá.
2. Un checkpoint `v` antiguo se puede leer como campo de flujo dividiendo por
   `k`, así que los solvers ODE se prueban sin reentrenar.
3. El Engineer no se toca: los estados `x_t`, el reparto de `t` y el embedding
   `999t` son idénticos.

Lo implementado vive en `src/tfm_shells/cosine_flow.py` con integradores ODE
(`euler`, `midpoint`, `heun`, `rk4`, `exponential`). Órdenes de convergencia
empíricos medidos contra campo oráculo: **1.00 / 1.98 / 1.98 / 3.98**, y el
exponencial en precisión de máquina (es exacto para esta trayectoria y coincide
con DDIM determinista).

### Trampa del offset `s`

Con `s = 0.008`, en `t = 0` la trayectoria **no llega a la geometría limpia**:
`α(0) = 0.99992`, **`σ(0) = 0.0124663`**. La solución exacta de la ODE en `t=0`
es `α(0)·z + σ(0)·ε`, o sea **~4,9 cm de ruido blanco por píxel** con la escala
del dataset (`z_scale = 3.9255`), que cae entero sobre la curvatura y por tanto
sobre la energía de flexión.

DDIM lo esconde porque su último paso devuelve `x̂₀` directamente. Un integrador
ODE fiel lo reproduce. Por eso `integrate_cosine_flow` lleva
`final_denoise=True`. **Sin esa proyección final, cambiar a un solver ODE empeora
los picos en vez de mejorarlos.**

Se detectó midiendo el orden de convergencia: los RK se estancaban en un error
de 2.88e-2 que no bajaba con más pasos, y ese número es exactamente
`σ(0)·max|ε|`.

### SNR: coseno frente a lineal

| t | SNR coseno / SNR lineal |
|---|---|
| 0.9 | 2.00 |
| 0.7 | 1.39 |
| 0.5 | 0.98 |
| 0.2 | 0.55 |
| 0.1 | 0.43 |

El coseno conserva el doble de señal en la mitad ruidosa y la pierde en la
limpia. La campana de guiado actúa en el 68% central. **El compromiso
integrabilidad (trayectorias rectas) frente a informatividad del gradiente
físico (trayectorias curvas de alto SNR) es específico del diseño guiado por
física y no está medido en la literatura.**

---

## 5. Hallazgo 2 — el muestreo determinista daña la métrica estructural

Observación del autor: con DDIM aparecen picos en la geometría; con DDPM no.

### Álgebra

Escribiendo ambos pasos en la misma forma, con `ε̂ = (x_t − √ᾱ_t·x̂₀)/√(1−ᾱ_t)`:

```
DDIM:  x_s = [α_s − σ_s·α_t/σ_t]·x̂₀  +  (σ_s/σ_t)·x_t
DDPM:  x_s = [α_s·β/σ_t²]·x̂₀         +  (α_t·σ_s²)/(α_s·σ_t²)·x_t  +  ω·n
```

(verificado contra `ddim_step` a 1e-13). El factor de arrastre del estado:

```
c_DDPM / c_DDIM = tan(φ_s)/tan(φ_t) < 1   siempre
```

DDPM arrastra estrictamente menos y rellena con ruido fresco. A `t` alto,
`c_DDIM ≈ 0.99`: DDIM conserva casi todo el estado.

**Pero un análisis lineal exacto por banda de frecuencia refuta que sea
amplificación lineal**: DDIM y DDPM amplifican un sesgo casi idéntico (DDPM
incluso algo más). El efecto es no lineal.

### Evidencia con los pesos reales (Architect epoch 45 del repo nuevo, 8 muestras)

Rugosidad = `|laplaciano|` del mapa de alturas, en centímetros:

| sampler | NFE | rugosidad media | **rugosidad max** | frac. alta frec |
|---|---|---|---|---|
| **láminas reales** | — | 2.99 | **30.68** | 0.00011 |
| ddim@50 | 50 | 3.19 | **64.90** | 0.00024 |
| ddpm@50 | 50 | 1.71 | **27.98** | 0.00018 |
| ddim@200 | 200 | 3.32 | **68.08** | 0.00024 |
| heun@25 | 49 | 3.96 | **72.22** | 0.00024 |
| rk4@13 | 49 | 4.36 | **70.45** | 0.00023 |
| euler@50 | 50 | 17.14 | 197.41 | 0.00147 |

**Concluyente: no es error de discretización.** `ddim@200` con 4× más pasos es
*peor* que `ddim@50`. Heun y RK4 a coste igualado también. Cuanto más exacta la
integración, más picos, porque convergen mejor a una trayectoria ODE que está
sesgada. Euler es directamente inservible.

### Mecanismo

Rastreando la rugosidad de `x̂₀` paso a paso, las dos trayectorias empiezan
idénticas y **divergen sobre el paso 19–27 (`t ≈ 0.38 → 0.21`)**: DDIM se queda
en 167–188 cm de máximo mientras DDPM cae a 96 y luego a 59.

En DDIM, `x_s = α_s·x̂₀ + σ_s·ε̂` deja el estado nuevo **exactamente sobre la
trayectoria definida por las propias estimaciones del modelo**. El modelo recibe
justo el estado que habría producido ese `x̂₀`, así que no tiene motivo para
cambiar de opinión: **es un punto fijo y un error se autoconfirma**. DDPM
sustituye parte de la dirección de ruido por un sorteo independiente, rompe esa
consistencia y obliga a re-decidir contra un prior que es suave.

En generación de imágenes esto no se nota. **En diseño de láminas la curvatura
es la física**, así que el artefacto de alta frecuencia aterriza directamente
sobre la energía de flexión, o sea sobre la métrica que se está optimizando.

### El eje útil es `eta`

Barrido a 50 pasos, mismo ruido inicial (Architect epoch 45, sin guiado):

| eta | rugosidad media | rugosidad max | veredicto |
|---|---|---|---|
| real | 2.99 | 30.68 | objetivo |
| 0.0 | 3.19 | 64.90 | picos |
| 0.2 | 3.09 | 48.35 | picos |
| **0.4** | **2.39** | **35.81** | **OK** |
| **0.6** | **2.16** | **32.47** | **OK** |
| 0.8 | 2.07 | 30.70 | suavizado |
| 1.0 | 1.77 | 28.11 | suavizado |

DDPM (`eta=1`) **sobre-suaviza**: pierde detalle legítimo (1.77 frente a 2.99
de una lámina real). El óptimo está en `eta ≈ 0.4–0.6`. Son 8 muestras de una
semilla: señal fuerte y monótona, no resultado calibrado.

**Confusión pendiente:** el sampler publicado corre con `clip_sample=True` *y*
ruido ancestral, o sea dos defensas simultáneas contra el artefacto. Hay que
cruzarlas para saber cuál trabaja.

---

## 6. Hallazgo 3 — calibración del evaluador FEM

Se decidió unificar toda la evaluación estructural en **Kratos** (script PEP 723
`evaluate_kratos.py`, entorno propio con KratosMultiphysics 10.4.3, sin torch).
Hipótesis físicas registradas: planta 10×10 m, espesor 0,10 m, E = 30 GPa,
ν = 0,20, densidad 2500 kg/m³, g = 9,81 m/s². Peso propio sobre el área real
inicial de cada elemento. Empotramiento de las seis componentes en nodos con
`z ≤ 0,10·max(z)`. Elemento `ShellThinElementCorotational3D4N` con
Newton–Raphson, análisis geométricamente no lineal.

Nota: el peso propio que aplica Kratos es `2500·0,1·9,81 = 2452,5 N/m²`, mientras
el `fz` del dataset es `24525 N/m²` — **factor 10**. En la práctica el cociente
de energías resulta poco sensible al nivel de carga en este régimen, pero
conviene tenerlo presente.

### Qué métrica reproduce Abaqus

Sobre 8 láminas reales que cubren MF de 0.41 a 0.95:

| Métrica | Sesgo | MAE | Correlación |
|---|---|---|---|
| `mf_mean` | +0.008 | 0.037 | 0.97 |
| `mf_resultants_area_mean` | +0.030 | 0.041 | 0.97 |

Pero **en el régimen que importa** (láminas generadas, MF ≈ 0.9), sobre las 40
geometrías verificadas en Abaqus del paper 1:

| γ | Abaqus (paper) | Kratos `mf_mean` | Kratos `mf_resultants_area_mean` |
|---|---|---|---|
| 10 | 0.913 | 0.876 (−0.037) | **0.9175 (+0.005)** |
| 250 | 0.921 | 0.889 (−0.032) | **0.9286 (+0.008)** |

**`mf_resultants_area_mean` reproduce Abaqus a ±0.008 y recupera la mejora
γ=10→250 (+0.011 frente a +0.008).** Es la métrica a usar. Tiene sentido: es la
fórmula de `Dataset_Kratos`, con la que se construyó el dataset.

Kratos confirma además de forma independiente el sesgo del surrogate: predice
0.978/0.981, el FEM dice 0.917/0.929.

---

## 7. Estructura de coste de inferencia

Medido en CPU, lote 2:

| Componente | Params | Coste por paso | Fracción |
|---|---|---|---|
| Architect (forward) | 120,2 M | 0,26 s | **11%** |
| Engineer PB-PUNet (**forward + backward**) | 360,8 M | 2,13 s | **89%** |

El guiado necesita el backward a través del Engineer en cada paso, y el Engineer
son tres UNets. **Cambiar el Architect mueve el 11% de la factura.** Los únicos
resortes reales son menos pasos y un surrogate más barato.

Alternativa disponible: `hybrid_fourier_unet`, **11,2 M** parámetros, tronco
compartido con mezclado espectral tipo FNO y tres cabezas, fwd+bwd **0,11 s**.
Es **32× menor y 19× más rápido** sobre el 89% del coste.

Medida en A100, lote 50, con los pesos del paper: **0,349 s por paso-lote**
(1 Architect + 1 Engineer fwd+bwd).

---

## 8. El paper 2

### Pregunta

> ¿Hasta dónde se puede abaratar la inferencia en diseño generativo guiado por
> física, y qué limita esa frontera?

Importa porque el valor de un prior generativo frente a la optimización
topológica es **explorar muchas alternativas**. A 40 min por lámina se ha
reconstruido el problema que el paper 1 criticaba en la introducción.

### Tesis

> El techo de velocidad no lo pone la precisión de la integración, sino la
> fiabilidad del surrogate a ruido alto y la estocasticidad del sampler.

Contraintuitivo, y ahí está el paper: el recetario estándar de aceleración
(samplers deterministas, solvers de orden alto, menos pasos) **falla aquí**,
con evidencia ya medida.

### Contribuciones

**C1 — el surrogate noise-aware *habilita* el muestreo rápido, no solo lo
mejora.** Con K=1000, que el surrogate sea fiable a ruido alto es marginal. Con
K=20, cada paso es un salto enorme por territorio de ruido alto, así que pasa a
ser la restricción activa. Predicción falsable: *la ventaja sobre Tweedie/DPS
crece al bajar K*. **No hay ningún dato aún: es la hipótesis que decide el
paper.**

**C2 — diagnóstico del fallo determinista, con mecanismo.** Sección 5.

**C3 — presupuesto de guía desacoplado del de muestreo.** El surrogate es el 89%
del coste pero la campana ya vale ≈0 fuera de una ventana: muestrear a K pasos y
guiar solo en M ≪ K. Más el surrogate espectral 32× menor.

**C4 — la geometría de la trayectoria interactúa con la guía física.** El
compromiso de SNR de la sección 4. Requiere reentrenar sobre interpolante
rectificado; fuera de la primera ronda.

### Qué NO reclamar

- "Pasamos de difusión a flow matching" — es el mismo modelo escalado por `k`.
- "Flow matching latente + guía física" como categoría — ya lo hacen PhysGen
  (CVPR 2026) y 3DID (NeurIPS 2025).
- El schedule en campana dentro de flow matching — van Delden et al.,
  *Minimizing Structural Vibrations via Guided Flow Matching*, ya usa un factor
  β dependiente del tiempo en forma de campana. Citar y diferenciarse.

### Posicionamiento

- **PBFM** (ICLR 2026) nombra el Jensen gap explícitamente pero lo resuelve con
  *unrolling* en entrenamiento + gradientes conflict-free, no con un surrogate
  noise-aware. Es el vecino teórico más próximo.
- **FlowDPS** (ICCV 2025) usa Tweedie sobre la estimación limpia.
- **D-Flow, OC-Flow, Dflow-SUR** evalúan el surrogate solo en la muestra limpia
  final.
- **PhysGen** aplica un surrogate *limpio* al latente ruidoso sin
  condicionamiento temporal, y mitiga con re-ruido/alternancia.

Ninguno entrena un surrogate condicionado al tiempo para la guía. Ese es el
hueco limpio.

---

## 9. Diseño experimental

### Decisiones fijadas

- **Solo Kratos.** No se reutilizan los resultados de Abaqus del paper 1; todas
  las muestras se generan de nuevo y se evalúan con una sola herramienta, para
  no tener que explicar dos pipelines FEM. **100 muestras por configuración**,
  no 20.
- **Métrica: `mf_resultants_area_mean`.**
- **La calidad nunca se lee del Engineer que da el gradiente** (sesgo +0.06).
- **Comparaciones pareadas**: un único pool de ruido inicial, sorteado una vez
  con semilla fija, reutilizado por todos los runs. El stream de ruido ancestral
  se siembra desde la posición en el pool, así los resultados no dependen del
  tamaño de lote.
- **Rejilla discreta de enteros.** El Architect se entrenó con timesteps enteros
  y el calendario de Diffusers, cuyo `alphas_cumprod` difiere del coseno
  analítico continuo hasta 5e-4, y nunca vio un timestep fraccionario.
- **La tasa de convergencia FEM es métrica de primera clase.** A K bajo el
  solver no converge y las muestras fallidas se excluyen de las medias: sin
  reportarla, la frontera saldría sesgada a favor de los samplers rápidos.

### Proveedores de gradiente

| Nombre | Dónde evalúa | Nota |
|---|---|---|
| `noise_aware` | `R(x_t, t)` | el Engineer del paper |
| `tweedie_clean` | `R_clean(x̂₀, 0)` | baseline DPS: surrogate limpio aparte sobre la estimación de Tweedie |
| `tweedie_self` | `R(x̂₀, 0)` | mismos pesos noise-aware sobre `x̂₀`: aísla *dónde evalúas* de *con qué entrenaste* |
| `naive_clean` | `R_clean(x_t, 0)` | surrogate limpio alimentado con el estado ruidoso |

En los Tweedie la salida del Architect va **detached** (forma práctica estándar
de DPS, y es lo que hacía el paper 1), así que cuestan 2 Architect + 1 surrogate
por paso.

### Matriz, 66 runs

| Bloque | Runs | Varía | Fija |
|---|---|---|---|
| **b1** frontera | 36 | provider{3} × K{10,20,50,100,250,1000} × eta{0,1} | clip=T, γ=250×8, pbunet |
| **b2** estocasticidad | 10 | eta{0,.25,.5,.75,1} × clip{T,F} | K=50, pbunet |
| **b3** escala de guía | 10 | K{20,1000} × γ{0,10,50,250,1000} | eta=1, clip=T, pbunet |
| **b4** surrogate barato | 6 | K{10,20,50,100,250,1000} | eta=1, clip=T, **hybrid** |
| **b5** guía dispersa | 4 | guide_every{1,2,4,10} | K=100, eta=1, pbunet |

C3 se lee cruzando b4 contra las 6 filas `noise_aware, eta1` de b1.

### Caveats conocidos de b4

- **Confusión de entrenamiento**: PB-PUNet epoch 51, híbrido epoch 35. Si el
  híbrido sale peor no se sabrá si es arquitectura o entrenamiento más corto.
- **Calendario**: el híbrido se entrenó con el coseno *continuo* del repo nuevo;
  el Architect del paper usa el discreto. Difieren ≤5e-4 en α.

---

## 10. La suite de código

`experiments/paper2/` en el repo `Flow_Matching`, más wrappers en la raíz
(`run_generation.py`, `run_kratos_eval.py`, `run_analysis.py`).

| Etapa | Dónde | Qué hace |
|---|---|---|
| `generate.py` | GPU (Colab A100) | produce todas las geometrías, registra coste. No juzga mecánica |
| `evaluate_all.py` | CPU local | lanza Kratos en paralelo por procesos |
| `analyze.py` | cualquiera | une coste + veredicto FEM + diversidad → `frontier.csv` |

Ambas etapas pesadas son reanudables (manifest incremental / `summary.json`).

Módulos: `discrete_vp.py` (sampler), `providers.py` (gradiente),
`models.py` (carga), `matrix.py` (matriz), `vae.py` (VAE auxiliar del paper 1,
solo encoder, para el hull latente).

### Validación del sampler

Un solo sampler cubre todo el eje de estocasticidad y es fiel en ambos extremos:

- `eta=1` + `clip=True` reproduce la **media posterior de DDPM a 1e-15** y su
  varianza a 1e-8 (el sampler del paper 1)
- `eta=0` + `clip=False` reproduce **DDIMScheduler a 1e-7**
- rejilla de timesteps idéntica a la de Diffusers para K ∈ {10,20,50,100,250,1000}

Detalle: Diffusers calcula `ε̂` **antes** de recortar `x̂₀` y no lo recalcula;
esta implementación sí lo recalcula, que es lo que hace `DDPMScheduler` vía la
fórmula de la media posterior. Por eso coincide exactamente con DDPM y difiere
de DDIM solo en el caso recortado.

### Detalle operativo

Los `run_id` descriptivos reventaban el **límite de 260 caracteres de ruta de
Windows** con el anidamiento `kratos/<run>/samples/sample_XXXX.json`. Se usa un
slug de 11 caracteres (`b1_c58ad09d`) en disco y el nombre legible en las tablas.

---

## 11. Estado y resultado preliminar

Prueba de la cadena completa con los checkpoints del paper, K=50, eta=1,
γ=250×8, clip=True, **2 muestras**:

```
FEM mf = 0.9356   convergencia 2/2   P>0.90 = 1.00   rugosidad 1.99 / 29.79 cm
```

Frente a las muestras de K=1000 del paper evaluadas con el mismo Kratos:
**0.9175** (γ=10) y **0.9286** (γ=250). O sea K=50 sale igual o mejor con 20×
menos pasos. **Son 2 muestras: anécdota alentadora, no resultado.**

### Expectativas antes del barrido

**Confianza alta:**
- La frontera será plana hasta K≈50 y se romperá sobre K=10.
- `eta=0` se verá bien en MF y mal en curvatura: mirar `roughness_max_cm`, no
  `fem_mf_mean`.
- A K=10 habrá fallos de convergencia FEM, sobre todo con `eta=0`. Vigilar
  `fem_n`.

**Desconocido, y decide el paper:** C1.

**Confianza media:** b4 aguanta; a K=20 probablemente haga falta γ mayor que a
K=1000.

**Riesgo real:** que todo salga plano — que K=50 iguale a K=1000 con los tres
proveedores por igual. Entonces queda un resultado práctico bueno ("20× más
barato") sin explicación de por qué, y C1 se cae. El paper pivotaría a C2+C3,
que ya están medidos.

### Lo que falta

- `clean_solid.pt` (el `modelo_clean` del paper 1) para `tweedie_clean` y
  `naive_clean`. **Sin él C1 no tiene con qué compararse.**
- Para C3 a prueba de revisores: reentrenar el híbrido 51 épocas sobre el
  calendario discreto.
- Para C4: entrenar sobre interpolante rectificado.

---

## 12. Requisitos para que el paper sea de calidad alta

1. **Verificación FEM en toda la frontera**, no en una muestra. Kratos calibrado
   lo hace viable para cientos de láminas por punto. Medir la calidad con el
   mismo surrogate que da el gradiente es circular.
2. **Diversidad en todo el barrido**, no solo MF: área del hull latente, DPP
   log-likelihood, CV(z_max). Un sampler rápido que colapsa la distribución no
   vale.
3. **Coste en NFE y segundos, desglosado.** El reparto 11%/89% justifica por qué
   se ataca el Engineer y no el Architect.
4. **Baselines reimplementados**: mínimo Tweedie/DPS y una de guía terminal.
   PBFM si se llega.
5. **Ablaciones que aíslen cada factor**: K, eta, tipo de surrogate, tamaño,
   trayectoria. Uno cada vez.
6. **Potencia estadística**: 100 muestras por configuración y varias semillas.
