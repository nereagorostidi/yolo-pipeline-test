from ultralytics import YOLO
import time

# Cargar la arquitectura ligera nano de YOLO11 (version mas reciente que YOLOv8)
model = YOLO('yolo11n.pt')

# Apuntamos la hora de inicio para poder medir cuánto tarda
inicio = time.time()
print('Empezando entrenamiento...')

# Ejecutar entrenamiento local
results = model.train(
    data='data.yaml',
    epochs=80,      # tope de pasadas completas al dataset; no tiene por que llegar a las 80
    patience=20,    # si en 20 epocas seguidas no mejora en validacion, para solo (evita overfitting)
    imgsz=640,
    batch=8,
    name='mi_modelo_frutas'  # si ya existe una carpeta con este nombre, YOLO crea mi_modelo_frutas-2, -3... (no sobreescribe)
)

print('¡Entrenamiento completado! El archivo resultante se ha guardado en:')
print('runs/detect/mi_modelo_frutas/weights/best.pt')

fin = time.time()
minutos = (fin - inicio) / 60
print(f'¡Entrenamiento completado en {minutos:.1f} minutos!')
print('El archivo resultante se ha guardado en:')
print('runs/detect/mi_modelo_frutas/weights/best.pt')