import cv2
from ultralytics import YOLO
model =YOLO('yolov8n.pt')
model.to('cpu')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
             print("No camera detected")
             break
    results = model(frame , stream  = True)
    for result in results:
       for box in result.boxes:
        conf = box.conf.item()

        if conf < 0.5:
            continue

        x1 , y1 , x2 , y2 = map(int , box.xyxy[0].tolist())
        name = model.names[int(box.cls.item())]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
        
