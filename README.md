# Taller YOLO — Detección de Frutas

Proyecto del taller de Inteligencia Artificial: un modelo de detección de objetos (YOLO) entrenado para reconocer 12 tipos de frutas en imágenes y vídeo.

## Contenido del repositorio

| Archivo | Qué hace |
|---|---|
| `data.yaml` | Configuración del dataset: rutas de train/valid/test y las clases de frutas. |
| `dataset_info.py` | Muestra por pantalla cuántas imágenes hay en train/valid y qué clases están configuradas. |
| `prueba1.py` | Entrena el modelo YOLO con el dataset de frutas. |
| `deteccion_video.py` | Aplica el modelo ya entrenado sobre un vídeo, muestra una ventana en vivo con las detecciones y guarda el resultado en `output.mp4`. |
| `runs/detect/` | Carpetas generadas automáticamente por cada entrenamiento (pesos del modelo, gráficas, métricas). |
| `train/`, `valid/`, `test/` | Imágenes y etiquetas del dataset (formato YOLO), descargadas de Roboflow. |

## El dataset

- 12 clases: `apple`, `bananas`, `grape`, `kiwifruit`, `lychee`, `mango`, `orange`, `peach`, `pear`, `pineapples`, `pomegranate`, `strawberry`.
- 276 imágenes de entrenamiento y 26 de validación (comprobado con `dataset_info.py`).

## Cómo entrenar el modelo

```bash
pip install ultralytics pyyaml opencv-python
python prueba1.py
```

`prueba1.py` parte de los pesos preentrenados `yolo11n.pt` y entrena hasta 80 épocas, con parada automática (`patience=20`) si el modelo deja de mejorar en validación. El resultado se guarda en `runs/detect/mi_modelo_frutas*/weights/best.pt` (cada ejecución crea una carpeta nueva para no pisar entrenamientos anteriores).

### Los entrenamientos guardados en `runs/detect/`

Las carpetas de resultados se nombran con el número de épocas con el que se entrenó cada vez:

| Carpeta | Épocas | Notas |
|---|---|---|
| `mi_modelo_frutas-15` | 15 | Primera prueba, entrenamiento corto. |
| `mi_modelo_frutas-80` | 80 (tope máximo, con `patience=20`) | El modelo que usa `deteccion_video.py` actualmente. Mejor punto real: época 74, con `mAP50-95 ≈ 0.847`. |

**¿Hacían falta 80 épocas?** No realmente. Mirando la curva de `mAP50-95` de ese entrenamiento, a partir de la época 40 el modelo se estabiliza: sigue oscilando entre 0.79 y 0.85 sin una mejora clara hasta el final. Es decir, con **`epochs=40`** ya se consigue prácticamente el mismo resultado, en la mitad de tiempo. Las 40 épocas extra no perjudican (no hay overfitting, la pérdida de validación no sube), simplemente no aportan mucho más.

**Limitación conocida:** el modelo confunde manzana (`apple`) y pera (`pear`) con cierta frecuencia — es un problema de dataset pequeño y clases visualmente parecidas, no un bug del código. Se ve reflejado en la matriz de confusión (`runs/detect/mi_modelo_frutas-80/confusion_matrix_normalized.png`).

## Cómo probarlo sobre un video

```bash
python deteccion_video.py
```

Al ejecutarlo, el script pregunta por terminal qué vídeo quieres analizar:

```
¿Como se llama el video a analizar? (ej. video_dron.mp4):
```

Escribe el nombre del archivo de vídeo (tiene que estar en esta misma carpeta) y pulsa Enter. Si el nombre no existe, te lo preguntará de nuevo, hasta que se encuentre el archivo correctamente.

Puedes usar tu propio vídeo (cópialo a la carpeta del proyecto primero) o probar con los dos de ejemplo que ya están subidos al repositorio:

- **`video_frutas.mp4`** — grabado con el móvil.
- **`video_dron.mp4`** — grabado con el dron.

En cualquier caso, el script analiza cada frame con el modelo entrenado, muestra una ventana en vivo con las cajas detectadas y guarda el vídeo anotado como `output.mp4`. Parámetros clave configurables dentro del script:

- `imgsz`: resolución de análisis (640 = misma que en el entrenamiento, más preciso).
- `conf`: confianza mínima para mostrar una detección.
- `vid_stride`: cuántos frames saltar (1 = analiza todos, más lento pero no se pierde nada).
- `augment`: test-time augmentation, mejora la precisión a cambio de velocidad.

Es un ajuste de velocidad vs. precisión: valores más altos de precisión (`imgsz` alto, `vid_stride=1`, `augment=True`) hacen el proceso más lento.
