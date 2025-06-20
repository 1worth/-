from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("last.pt")

train_results = model.train(
    data="data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="CPU",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)