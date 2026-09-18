# Difusión VP + DDIM para láminas sólidas

Esta carpeta replica el pipeline principal de `TFM` para láminas **solid** de 64 × 64. Conserva el Engineer `parallel_pb_unet` de tres ramas y 13 campos físicos. Architect e Engineer usan el mismo proceso de difusión VP coseno del script `../Code/Prueba/diffusion_exotic.py`. El Architect predice `v` y genera con DDIM.

## Método

- Se normaliza `z` a `[-1,1]` con estadísticas del train y se toma `ε ~ N(0,I)`.
- El calendario coseno continuo define `φ(t)=(t+s)/(1+s)·π/2`, `α=cos φ` y `σ=sin φ`; `x_t=α z+σ ε` y el objetivo del Architect es `v=α ε−σ z`. `t=0` representa geometría casi limpia y `t=1` ruido puro. El tiempo se muestrea de forma estratificada por lote.
- Se conserva la arquitectura `UNet2DModel` original de un canal. Su embedding temporal recibe `999t`. El muestreo DDIM recorre `t=1→0`, con `eta=0` por defecto, malla cuadrática y pesos EMA del Architect.
- El Engineer de tres UNet aprende **sobre los mismos estados VP y el mismo calendario** que el Architect: recibe `(x_t, fz)` y `999t`, y predice `uz`, seis campos de membrana y seis de flexión. Su pérdida supervisada da un tercio del peso a cada rama: `MSE_uz/3 + MSE_membrana/3 + MSE_flexión/3`, con canales normalizados usando solo el conjunto de entrenamiento. El residuo físico global recibe peso `(1−t)^p`, porque la geometría limpia está en `t=0`.
- La pérdida constitutiva opcional, activada en `configs/engineer.yaml`, compara los esfuerzos de membrana con `A ε` y los momentos con `D κ` para material elástico isótropo. Sus residuos se dividen por la desviación típica de cada salida FEM antes de elevar al cuadrado; membrana y flexión reciben igual peso. `E=30 GPa`, `ν=0,2` y `h=0,1 m` coinciden con los datos sólidos revisados. Las componentes `12` llevan el factor `1/2` de la convención del dataset. Ajusta estos parámetros si cambias de material, espesor o convención FEM.
- La evaluación del Engineer sobre la geometría limpia usa `t=0` tanto para la figura como para `mf_mae`.
- El guiado evalúa directamente el Engineer en el mismo estado VP del Architect. Modifica la predicción `v` con `+σ(t)·clip(γ w(1−t)∇(1−MF)^2, ±grad_clip)` antes del paso DDIM; el signo produce descenso del objetivo en la estimación de la muestra limpia. `γ` debe ajustarse experimentalmente.

El Architect no usa `fz` como entrada. La carga actúa en el Engineer y en el guiado, igual que en el pipeline original sin condición tipológica. Ambos checkpoints deben indicar el mismo `cosine_s`; los antiguos checkpoints de Flow Matching lineal y DDPM se rechazan. Hay que volver a entrenar ambos modelos. El *Stable Target Field* del ejemplo bidimensional no se traslada a imágenes 64 × 64: el Architect usa el objetivo `v` estándar. La codificación espacial de Fourier del MLP 2D tampoco se traslada literalmente; aquí se conserva la UNet de imágenes.

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

Las comprobaciones locales se ejecutan con `uv run python -m unittest discover -s tests -v`.

## Comprobar la convergencia en menos pasos

Tras entrenar ambos modelos:

```powershell
uv run python benchmark_steps.py --config configs/sample_guided.yaml --steps 10 20 50 100 250 --reference-steps 1000
```

Esto genera `artifacts/step_benchmark/step_benchmark.csv` y `metadata.json`. El benchmark exige `eta=0` y usa **el mismo ruido inicial**, calendario, `γ` y modelos en cada número de pasos. Se reportan tiempo, número de evaluaciones, MF medio, proporción con MF >0.90, diferencia de MF por muestra y MAE/RMSE geométrico frente a la trayectoria de 1000 pasos. Para comparar calidad estructural con el TFM original, evalúa además las muestras con Abaqus o un juez externo: la coincidencia con el propio surrogate no demuestra equivalencia física.

## Alcance

Esta copia no incluye el dataset ni los pesos entrenados. No hay resultados de convergencia nuevos hasta entrenar y ejecutar el benchmark. Los scripts de análisis históricos de `TFM` no son necesarios para lanzar este pipeline de sólidos.
