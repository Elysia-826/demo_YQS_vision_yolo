from ultralytics import  YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="",
    epochs = 50,
    imgsz = 640,
    batch = 8,
    patience = 10,
    project = 'runs/temp_seed',
    name = 'v1'
)
