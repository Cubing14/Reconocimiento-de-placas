import cv2
from ultralytics import YOLO
import easyocr

# Cargar modelo YOLO
modelo = YOLO("yolov8n.pt")

# Inicializar EasyOCR
lector = easyocr.Reader(["es"])  # 'es' para español, puedes agregar más idiomas como 'en'

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))  # Redimensiona la imagen para acelerar el procesamiento

    if not ret:
        break

    # Realizar detección con YOLO
    resultados = modelo(frame)

    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0])  # Coordenadas de la caja
            conf = caja.conf[0].item()  # Confianza de detección
            clas = caja.cls[0].item()   # Clase detectada

            if conf > 0.5:  # Filtrar detecciones con confianza alta
                # Extraer la región de la matrícula
                placa = frame[y1:y2, x1:x2]

                # Convertir a escala de grises para mejorar OCR
                placa_gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

                # Leer el texto de la placa con OCR
                resultado_ocr = lector.readtext(placa_gris)

                texto_placa = ""
                for (caja, texto, probabilidad) in resultado_ocr:
                    if probabilidad > 0.5:  # Filtrar textos con alta confianza
                        texto_placa += texto + " "

                # Dibujar la detección y el texto
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, texto_placa.strip(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar resultado
    cv2.imshow("Detección de Placas", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
exit()