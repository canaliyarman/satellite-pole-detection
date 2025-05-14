from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/obb/train/weights/last.pt")  # build from YAML and transfer weights

    results = model.train(resume=True)