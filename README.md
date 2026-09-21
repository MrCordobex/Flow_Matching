# Difusión VP + DDIM para láminas sólidas

Esta carpeta replica el pipeline principal de `TFM` para láminas **solid** de 64 × 64. El Engineer predeterminado es `hybrid_fourier_unet`: un tronco compartido con convoluciones locales, mezclado espectral y tres cabezas para los 13 campos físicos. Architect e Engineer usan el mismo proceso de difusión VP coseno del script `../Code/Prueba/diffusion_exotic.py`. El Architect predice `v` y genera con DDIM.

## Método

- Se normaliza `z` a `[-1,1]` con estadísticas del train y se toma `ε ~ N(0,I)`.
- El calendario coseno continuo define `φ(t)=(t+s)/(1+s)·π/2`, `α=cos φ` y `σ=sin φ`; `x_t=α z+σ ε` y el objetivo del Architect es `v=α ε−σ z`. `t=0` representa geometría casi limpia y `t=1` ruido puro. El tiempo se muestrea de forma estratificada por lote.
- Se conserva la arquitectura `UNet2DModel` original de un canal. Su embedding temporal recibe `999t`. El muestreo DDIM recorre `t=1→0`, con `eta=0` por defecto, malla cuadrática y pesos EMA del Architect.
- El Engineer aprende **sobre los mismos estados VP y el mismo calendario** que el Architect: recibe `(x_t, fz)` y `999t`, y predice `uz`, seis campos de membrana y seis de flexión. Su tronco U-Net comparte información entre las ramas: los bloques de Fourier mezclan información global a resoluciones 32 × 32 y 16 × 16, mientras las convoluciones 3 × 3 y las conexiones entre escalas conservan detalles locales. Se añaden coordenadas normalizadas y un embedding temporal continuo. El FFT usa un halo replicado para reducir la discontinuidad periódica de los bordes. Esta combinación está inspirada en [FNO](https://arxiv.org/abs/2010.08895) y [U-FNO](https://arxiv.org/abs/2109.03697); no supone que sus resultados en otros PDE se reproduzcan en estas láminas.
- La pérdida supervisada da un tercio del peso a cada rama: `MSE_uz/3 + MSE_membrana/3 + MSE_flexión/3`, con canales normalizados usando solo el conjunto de entrenamiento. Una pérdida de diferencias espaciales, igualmente equilibrada por ramas, supervisa la estructura local. Ambas pérdidas se calculan contra los campos FEM. El residuo físico global recibe peso `(1−t)^p`, porque la geometría limpia está en `t=0`.
- La pérdida constitutiva opcional, activada en `configs/engineer.yaml`, compara los esfuerzos de membrana con `A ε` y los momentos con `D κ` para material elástico isótropo. Sus residuos se dividen por la desviación típica de cada salida FEM antes de elevar al cuadrado; membrana y flexión reciben igual peso. `E=30 GPa`, `ν=0,2` y `h=0,1 m` coinciden con los datos sólidos revisados. Las componentes `12` llevan el factor `1/2` de la convención del dataset. Ajusta estos parámetros si cambias de material, espesor o convención FEM.
- La evaluación del Engineer sobre la geometría limpia usa `t=0` tanto para la figura como para `mf_mae`.
- El guiado evalúa directamente el Engineer en el mismo estado VP del Architect. Modifica la predicción `v` con `+σ(t)·clip(γ w(1−t)∇(1−MF)^2, ±grad_clip)` antes del paso DDIM; el signo produce descenso del objetivo en la estimación de la muestra limpia. `γ` debe ajustarse experimentalmente.

El Architect no usa `fz` como entrada. La carga actúa en el Engineer y en el guiado, igual que en el pipeline original sin condición tipológica. Ambos checkpoints deben indicar el mismo `cosine_s`; los antiguos checkpoints de Flow Matching lineal y DDPM se rechazan. Hay que entrenar el nuevo Engineer; los pesos de la PBUNet anterior no cargan en la nueva arquitectura. El *Stable Target Field* del ejemplo bidimensional no se traslada a imágenes 64 × 64: el Architect usa el objetivo `v` estándar.

## Datos y requisitos

Los YAML esperan el dataset en `../Datos/processed_image2/processed_image2`. Debe contener los `.npz` sólidos `shell_*.npz` con `z`, `fz`, `mf`, `ds`, `dv` y los 13 campos físicos. La carpeta `Datos` y los checkpoints no forman parte de esta copia. Ajusta `data.dataset_dir` y `conditioning.source_file` si están en otra ruta. Se usa Python 3.12 (fijado en `.python-version`) y las dependencias de `pyproject.toml`.

```powershell
cd Flow_Matching
uv sync
```

## Ejecutar como antes

```powershell
uv run python train_architect.py --config configs/architect.yaml
uv run python train_engineer.py --config configs/engineer.yaml
uv run python sample_guided.py --config configs/sample_guided.yaml
```

La CLI equivalente es `uv run tfm-shells architect|engineer|sample --config ...`. Los checkpoints se guardan en `models/architect/latest/best.pt` y `models/engineer/latest/best.pt`; las muestras en `artifacts/sample/<fecha>/guided_samples.npz`, con clave `z` y forma `(B,1,64,64)`. Los runs también registran métricas en MLflow.

El muestreo por defecto usa 50 pasos DDIM, `eta=0`, malla `quadratic` y `γ=1`; son valores iniciales para probar, **no resultados calibrados**. Cambia `sampling.num_inference_steps`, `sampling.time_spacing` (`uniform` o `quadratic`), `sampling.eta` y `sampling.guidance_scale` en el YAML. DDIM hace una evaluación del Architect por paso y, con guiado, una del Engineer.

Las comprobaciones locales se ejecutan con `uv run python -m unittest discover -s tests -v`. La arquitectura y los pesos de las pérdidas son hipótesis para probar. Compara `val_uz_mse`, `val_membrane_mse`, `val_flexion_mse`, `val_mf_mae` y la evaluación FEM externa con la PBUNet anterior antes de concluir que mejora la predicción. La consistencia constitutiva sí se comprobó sobre campos FEM; la pérdida global de energía no equivale a imponer todo el equilibrio de la lámina.

## Comprobar la convergencia en menos pasos

Tras entrenar ambos modelos:

```powershell
uv run python benchmark_steps.py --config configs/sample_guided.yaml --steps 10 20 50 100 250 --reference-steps 1000
```

Esto genera `artifacts/step_benchmark/step_benchmark.csv` y `metadata.json`. El benchmark exige `eta=0` y usa **el mismo ruido inicial**, calendario, `γ` y modelos en cada número de pasos. Se reportan tiempo, número de evaluaciones, MF medio, proporción con MF >0.90, diferencia de MF por muestra y MAE/RMSE geométrico frente a la trayectoria de 1000 pasos. Para comparar calidad estructural con el TFM original, evalúa además las muestras con Abaqus o un juez externo: la coincidencia con el propio surrogate no demuestra equivalencia física.

## Alcance

Esta copia no incluye el dataset ni los pesos entrenados. No hay resultados de convergencia nuevos hasta entrenar y ejecutar el benchmark. Los scripts de análisis históricos de `TFM` no son necesarios para lanzar este pipeline de sólidos.

## Evaluar el MF de las muestras con Kratos

Desde la raíz del repositorio, después del sampling:

```bash
uv run evaluate_kratos.py artifacts/sample/NOMBRE_DEL_RUN
# También acepta solo NOMBRE_DEL_RUN o la ruta directa al NPZ.
# Prueba de dos muestras:
uv run evaluate_kratos.py artifacts/sample/NOMBRE_DEL_RUN --limit 2
```

El script declara sus dependencias: `uv` prepara un entorno independiente con
Python 3.12, Kratos 10.4.3 y Plotly. No requiere torch, checkpoints ni la carpeta
local `Code/Prueba Kratos`. La primera ejecución necesita descargar dependencias.
Usar `uv run evaluate_kratos.py`, sin intercalar `python`, para activar esta instalación.

Lee `guided_samples.npz`, clave `z`, en metros, sin renormalizar, suavizar ni
desplazar la geometría. Admite `(B,1,H,W)`, `(B,H,W)` y `(H,W)`.
Cada píxel es un nodo; se utilizan cuadriláteros entre píxeles vecinos.
Solo sirve para láminas sólidas: no interpreta huecos a partir de ceros.

**Hipótesis físicas registradas en cada evaluación:** planta 10×10 m,
espesor 0,10 m, E=30 GPa, nu=0,20, densidad 2500 kg/m³ y gravedad 9,81 m/s².
Se aplica peso propio sobre el área real inicial de cada elemento, repartido
entre sus cuatro nodos. No se reutiliza el mapa `fz` del Engineer.
Se empotran las seis componentes de los nodos con **z <= 0,10 * max(z)**,
incluidos nodos interiores. El umbral se mide respecto a z=0, sin restar min(z).
El elemento `ShellThinElementCorotational3D4N` y Newton–Raphson siguen el
análisis geométricamente no lineal de `Prueba Kratos/generate_funicular_dataset.py`.
Esto es una evaluación FEM bajo estas hipótesis, no una reproducción idéntica
del S4R lineal de Abaqus ni una solución analítica exacta.

Las propiedades se pueden cambiar mediante `--span-x`, `--span-y`, `--thickness`,
`--young-modulus`, `--poisson-ratio`, `--density`, `--gravity` y `--support-fraction`.
El script comprueba convergencia, valores finitos y equilibrio de reacciones.
Las muestras que fallan se registran como `failed`, se excluyen de las medias y
el comando devuelve un código de error, conservando los resultados correctos.

### Qué significa cada MF

| Campo | Definición |
|---|---|
| `mf_mean` (principal) | Método de `Prueba Kratos`: cociente de energías nativas de membrana y flexión por elemento; promedio de elementos adyacentes a cada nodo y media de todos los nodos, incluidos apoyos. |
| `mf_energy_ratio` | Suma de las medidas nativas de energía de membrana dividida entre suma de membrana y flexión. No es una media espacial de MF. |
| `mf_resultants_area_mean` | Fórmula de `Dataset_Kratos`: energías reconstruidas a partir de N y M medios en los puntos de integración, promedio por área y exclusión de elementos con cuatro nodos empotrados. |

Para reproducir el primer método, se suman los valores no negativos que Kratos
devuelve para `SHELL_ELEMENT_MEMBRANE_ENERGY` y `SHELL_ELEMENT_BENDING_ENERGY`.
MF local = Em / max(Em + Eb, 1e-30). Se guarda el mapa nodal como `mf`.
No se mezclan las tres métricas ni se presentan como intercambiables.
El criterio de apoyos y el método de agregación también deben coincidir cuando
se compare con el MF estimado por el Engineer.

### Salidas y Jupyter/Colab

La subcarpeta `kratos/` del run contiene:

- `metrics.csv`, `summary.json` y `evaluation.json` con configuración y huella de entrada.
- `samples/sample_XXXX.npz`: geometría, apoyos, desplazamientos, reacciones,
  fuerzas nodales, áreas, N, M, energías y mapas de MF; JSON individual con métricas.
- `mf_gallery.html`: visor 3D autónomo, sin conexión, con selector de muestra,
  geometría coloreada por MF y apoyos negros; permite girar, ampliar y consultar valores.
- `mf_gallery.plotly.json` y `view_mf.ipynb`, para visualizar el resultado en Jupyter.

En el notebook donde ya has ejecutado el sampling:

```python
SAMPLING_DIR = '/content/Flow_Matching/artifacts/sample/NOMBRE_DEL_RUN'
!uv run evaluate_kratos.py "{SAMPLING_DIR}"
```

Después, para ver la figura dentro de una celda sin instalar Kratos en el kernel:

```python
from pathlib import Path
import html
from IPython.display import HTML, display

document = (Path(SAMPLING_DIR) / 'kratos' / 'mf_gallery.html').read_text(encoding='utf-8')
display(HTML('<iframe style="width:100%;height:780px;border:0" srcdoc="'
             + html.escape(document, quote=True) + '"></iframe>'))
```

También puedes abrir `kratos/view_mf.ipynb` junto a su HTML; si lo mueves,
ajusta `OUTPUT` en su celda. Volver a ejecutar el mismo comando reanuda los
casos pendientes; `--overwrite` recalcula los seleccionados. Para otra
configuración, entrada o versión del evaluador usa `--output-dir` distinto,
evitando mezclar resultados. `--start` y `--limit` permiten evaluar por bloques.
