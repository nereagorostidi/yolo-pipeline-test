# Taller YOLO — Detección de Frutas

Proyecto del taller de Inteligencia Artificial: un modelo de detección de objetos (YOLO) entrenado para reconocer 12 tipos de frutas en imágenes y vídeo.

## Contenido del repositorio

| Archivo | Qué hace |
|---|---|
| `data.yaml` | Configuración del dataset: rutas de train/valid/test y las clases de frutas. |
| `dataset_info.py` | Muestra por pantalla cuántas imágenes hay en train/valid y qué clases están configuradas. |
| `entrena.py` | Entrena el modelo YOLO con el dataset de frutas. |
| `deteccion_video.py` | Aplica el modelo ya entrenado sobre un vídeo, muestra una ventana en vivo con las detecciones y guarda el resultado en `output/`. |
| `samples/` | Vídeos de ejemplo (`video_frutas.mp4`, `video_dron.mp4`) para probar `deteccion_video.py`. |
| `output/` | Vídeos anotados generados por `deteccion_video.py`, con el mismo nombre que el vídeo de origen. |
| `runs/detect/` | Carpetas generadas automáticamente por cada entrenamiento (pesos del modelo, gráficas, métricas). |
| `train/`, `valid/`, `test/` | Imágenes y etiquetas del dataset (formato YOLO), descargadas de Roboflow. |

## El dataset

- 12 clases: `apple`, `bananas`, `grape`, `kiwifruit`, `lychee`, `mango`, `orange`, `peach`, `pear`, `pineapples`, `pomegranate`, `strawberry`.
- 276 imágenes de entrenamiento y 26 de validación (comprobado con `dataset_info.py`).

## Cómo entrenar el modelo

```bash
pip install ultralytics pyyaml opencv-python
python entrena.py
```

`entrena.py` parte de los pesos preentrenados `yolo11n.pt` y entrena hasta 80 épocas, con parada automática (`patience=20`) si el modelo deja de mejorar en validación. El resultado se guarda en `runs/detect/mi_modelo_frutas*/weights/best.pt` (cada ejecución crea una carpeta nueva para no pisar entrenamientos anteriores).

### Los entrenamientos guardados en `runs/detect/`

Las carpetas de resultados se nombran con el número de épocas con el que se entrenó cada vez:

| Carpeta | Épocas | Notas |
|---|---|---|
| `mi_modelo_frutas-15` | 15 | Primera prueba, entrenamiento corto. |
| `mi_modelo_frutas-80` | 80 (tope máximo, con `patience=20`) | El modelo que usa `deteccion_video.py` actualmente. Mejor punto real: época 74, con `mAP50-95 ≈ 0.847`. |

**¿Hacían falta 80 épocas?** No realmente. Mirando la curva de `mAP50-95` de ese entrenamiento, a partir de la época 40 el modelo se estabiliza: sigue oscilando entre 0.79 y 0.85 sin una mejora clara hasta el final. Es decir, con **`epochs=40`** ya se consigue prácticamente el mismo resultado, en la mitad de tiempo. Las 40 épocas extra no perjudican (no hay overfitting, la pérdida de validación no sube), simplemente no aportan mucho más.

**Limitación conocida:** el modelo confunde manzana (`apple`) y pera (`pear`) con cierta frecuencia — es un problema de dataset pequeño y clases visualmente parecidas, no un bug del código. Se ve reflejado en la matriz de confusión (`runs/detect/mi_modelo_frutas-80/confusion_matrix_normalized.png`).

## Cómo probarlo sobre un video

El nombre del vídeo se pasa como parámetro por línea de comandos:

```bash
python deteccion_video.py samples/video_dron.mp4
```

Puedes usar tu propio vídeo (cópialo a la carpeta del proyecto, o a `samples/`, y usa esa ruta) o probar con los dos de ejemplo que ya están subidos al repositorio dentro de `samples/`:

- **`samples/video_frutas.mp4`** — grabado con el móvil.
- **`samples/video_dron.mp4`** — grabado con el dron.

El script analiza cada frame con el modelo entrenado, muestra una ventana en vivo con las cajas detectadas y guarda el vídeo anotado en la carpeta `output/`, con el mismo nombre que el vídeo original (por ejemplo, `video_dron.mp4` → `output/video_dron.mp4`).

### Parámetros opcionales

Además del vídeo (obligatorio), se pueden ajustar por línea de comandos:

| Parámetro | Qué hace | Por defecto |
|---|---|---|
| `--conf` | Confianza mínima para mostrar una detección (subir = menos falsos positivos, bajar = menos frutas sin detectar). | `0.5` |
| `--vid-stride` | Cuántos frames saltar (1 = analiza todos, más lento pero no se pierde nada). | `3` |
| `--augment` / `--no-augment` | Test-time augmentation: analiza cada frame varias veces (flips/escalas) y combina resultados, mejora la precisión a cambio de velocidad. | activado |

Ejemplo cambiando varios a la vez:

```bash
python deteccion_video.py samples/video_dron.mp4 --conf 0.6 --vid-stride 1 --no-augment
```

Si no se indica ningún parámetro opcional, se usan los valores por defecto de la tabla. Para ver la ayuda completa con todos los parámetros disponibles y sus valores por defecto:

```bash
python deteccion_video.py --help
```

Es un ajuste de velocidad vs. precisión: valores más altos de precisión (`vid_stride=1`, `--augment`) hacen el proceso más lento. La resolución de análisis (`imgsz=640`, misma que en el entrenamiento) no es configurable por parámetro, se fija dentro del script.
