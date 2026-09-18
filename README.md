# Flow Matching para láminas sólidas

Esta carpeta replica el pipeline principal de `TFM` para láminas **solid** de 64 × 64. Conserva el Engineer `parallel_pb_unet` de tres ramas y 13 campos físicos. El Architect usa *linear conditional flow matching* en lugar de DDPM. Las pruebas de `Code/Prueba` sirven de referencia para la formulación `x_t=(1-t)ε+t z`, objetivo `z-ε` e integración de `dx/dt=vθ(x,t)`.

## Método

- Se normaliza `z` a `[-1,1]` con estadísticas del train y se toma `ε ~ N(0,I)`.
- Por muestra se elige `t ~ U(0,1)`, se construye `x_t=(1-t)ε+t z` y se entrena el Architect para predecir `v=z-ε` mediante MSE.
- Se conserva la arquitectura `UNet2DModel` original de un canal. Su embedding temporal recibe `999t`; este escalado está guardado en el checkpoint.
- El Engineer de tres UNet recibe `(x_t, fz)` y `999t`. Predice `uz`, seis campos de membrana y seis de flexión. Su pérdida supervisada da un tercio del peso a cada rama: `MSE_uz/3 + MSE_membrana/3 + MSE_flexión/3`, con canales normalizados usando solo el conjunto de entrenamiento. El residuo físico global conserva el peso temporal `t^p`, porque `t=1` es la geometría limpia.
- La pérdida constitutiva opcional, activada en `configs/engineer.yaml`, compara los esfuerzos de membrana con `A ε` y los momentos con `D κ` para material elástico isótropo. Sus residuos se dividen por la desviación típica de cada salida FEM antes de elevar al cuadrado; membrana y flexión reciben igual peso. `E=30 GPa`, `ν=0,2` y `h=0,1 m` coinciden con los datos sólidos revisados. Las componentes `12` llevan el factor `1/2` de la convención del dataset. Ajusta estos parámetros si cambias de material, espesor o convención FEM.
- En Flow Matching, la evaluación del Engineer sobre la geometría limpia usa `t=1` tanto para la figura como para `mf_mae`.
- El sampler integra desde `t=0` hasta `t=1` con Euler o Heun. El guiado usa `v_guiada = vθ - clip(γ w(t) ∇(1-MF)^2, ±grad_clip)`. El valor `γ` **no es directamente comparable** con el de DDPM; debe ajustarse experimentalmente.

El Architect no usa `fz` como entrada. La carga actúa en el Engineer y en el guiado, igual que en el pipeline original sin condición tipológica. Los checkpoints DDPM originales no se aceptan porque su parametrización temporal y sus estados ruidosos son distintos.

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

El muestreo por defecto usa 50 pasos Euler y `γ=1`; son valores iniciales para probar, **no resultados calibrados**. Cambia `sampling.num_inference_steps`, `sampling.solver` (`euler` o `heun`) y `sampling.guidance_scale` en el YAML. Heun hace dos evaluaciones del Architect por paso; con guiado hace también dos del Engineer.

Las comprobaciones locales se ejecutan con `uv run python -m unittest discover -s tests -v`.

## Comprobar la convergencia en menos pasos

Tras entrenar ambos modelos:

```powershell
uv run python benchmark_steps.py --config configs/sample_guided.yaml --steps 10 20 50 100 250 --reference-steps 1000
```

Esto genera `artifacts/step_benchmark/step_benchmark.csv` y `metadata.json`. Cada número de pasos usa **el mismo ruido inicial**, el mismo solver, el mismo `γ` y los mismos modelos. Se reportan tiempo, número de evaluaciones, MF medio, proporción con MF >0.90, diferencia de MF por muestra y MAE/RMSE geométrico frente a la trayectoria de 1000 pasos. Para comparar calidad estructural con el TFM original, evalúa además las muestras con Abaqus o un juez externo: la coincidencia con el propio surrogate no demuestra equivalencia física.

## Alcance

Esta copia no incluye el dataset ni los pesos entrenados. No hay resultados de convergencia nuevos hasta entrenar y ejecutar el benchmark. Los scripts de análisis históricos de `TFM` no son necesarios para lanzar este pipeline de sólidos.
