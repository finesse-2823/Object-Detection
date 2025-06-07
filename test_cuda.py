import torch
import cv2
from ultralytics import YOLO

print("=== CUDA Installation Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test YOLO with CUDA
    print("\n=== Testing YOLO with CUDA ===")
    try:
        model = YOLO('yolov8s.pt')
        model.to('cuda')
        print("✓ YOLO model loaded successfully on GPU")
        
        # Test inference speed
        import time
        dummy_input = torch.randn(1, 3, 640, 640).cuda()
        start_time = time.time()
        with torch.no_grad():
            _ = model.model(dummy_input)
        inference_time = time.time() - start_time
        print(f"✓ GPU inference time: {inference_time*1000:.1f}ms")
        
    except Exception as e:
        print(f"✗ Error loading YOLO: {e}")
else:
    print("✗ CUDA not available - check installation")

print("\n=== OpenCV Test ===")
print(f"OpenCV version: {cv2.__version__}")
print("✓ OpenCV working")