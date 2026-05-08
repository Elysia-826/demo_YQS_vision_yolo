from ultralytics import  YOLO

model_path = r"C:\Users\YQS\Desktop\yolo\yolo11n.pt"
model = YOLO(model_path)
results = model.train(
    data="./dataset/data.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 8,
    device = 0,
    workers = 0,
    patience = 10,
    project = 'runs/temp_seed',
    name = 'v1'
)