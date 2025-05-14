from thop import profile
from ultralytics import YOLO
import torch
if __name__ == '__main__':
    model = YOLO("./runs/obb/train/weights/last.pt")  # build from YAML and transfer weights

    #results = model.train(resume=True)
    # Assuming you have an input image size, e.g., 640x640
    input_size = (1, 3, 1024, 1024)  # Batch size 1, 3 color channels, 640x640 resolution

    # Create a dummy input tensor
    dummy_input = torch.randn(input_size)

    # Fuse the model for inference (optional but recommended for accuracy)
    model.fuse()

    # Calculate the number of parameters and FLOPs
    flops, params = profile(model.model, inputs=(dummy_input, ))

    # Convert to GFLOPs
    gflops = flops / 1e9

    print(f"GFLOPs: {gflops:.2f}")
    print(f"Number of parameters: {params}")