from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
# Load a model
    model = YOLO("best.pt")  # load a pretrained model (recommended for training)

    # Train the model with 2 GPUs
    results = model.train(data="etdii.yaml", epochs=20, imgsz=1024, device=0)