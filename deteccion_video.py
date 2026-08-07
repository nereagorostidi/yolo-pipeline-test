import cv2
from ultralytics import YOLO

# Cargar VUESTRO cerebro entrenado
model = YOLO('runs/detect/mi_modelo_frutas-80/weights/best.pt')

VID_STRIDE = 3  # solo procesa 1 de cada 3 frames, para ir mas rapido

# fps original del video, para que el output.mp4 dure lo mismo que el original
cap_info = cv2.VideoCapture('video_dron.mp4')
fps_original = cap_info.get(cv2.CAP_PROP_FPS)
cap_info.release()

# stream=True: procesa el video frame a frame (con el salto de vid_stride)
# sin cargarlo entero en memoria
results = model.predict(
    source='video_dron.mp4',
    imgsz=640,       # igual que el imgsz de entrenamiento; a menos resolucion se pierde detalle y confunde mas las clases
    conf=0.5,        # confianza minima para mostrar una deteccion (subir = menos falsos positivos, bajar = menos frutas sin detectar)
    vid_stride=1,    # 1 = analiza todos los frames; subirlo va mas rapido pero puede saltarse frutas que pasan rapido
    stream=True,
    verbose=False,
    augment=True     # test-time augmentation: analiza cada frame varias veces (flips/escalas) y combina resultados, mas preciso pero mas lento
)

writer = None
for r in results:
    annotated_frame = r.plot()

    if writer is None:
        h, w = annotated_frame.shape[:2]
        writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_original / VID_STRIDE,
            (w, h),
        )
    writer.write(annotated_frame)

    # Redimensionar para la vista previa manteniendo la proporcion original
    # (960x540 fijo deformaba los videos verticales del iPhone)
    h, w = annotated_frame.shape[:2]
    escala = 540 / h
    preview = cv2.resize(annotated_frame, (round(w * escala), 540))

    # Mostrar ventana interactiva
    cv2.imshow('Detector de Frutas YOLO', preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cv2.destroyAllWindows()
