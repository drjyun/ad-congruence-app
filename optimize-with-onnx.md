# 🔥 Quick Optimization: ONNX Runtime

Make your app 2-3x faster by using ONNX Runtime instead of PyTorch.

## Why ONNX Runtime?

- ✅ 2-3x faster inference on CPU
- ✅ Stay in Python (no Java needed!)
- ✅ Same model quality
- ✅ Lower memory usage
- ✅ Cross-platform

## Implementation (1-2 hours)

### Step 1: Export ViT to ONNX

```python
import torch
from transformers import ViTModel, ViTImageProcessor

# Load model
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "vit_base.onnx",
    input_names=['pixel_values'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'}
    },
    opset_version=14
)
```

### Step 2: Update app.py to use ONNX

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
ort_session = ort.InferenceSession(
    "vit_base.onnx",
    providers=['CPUExecutionProvider']
)

def vit_embeddings_onnx(frames_dir):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    feats = []
    
    for frame_path in frames:
        img = Image.open(frame_path).convert("RGB")
        img = transform(img).unsqueeze(0).numpy()
        
        # Run ONNX inference
        outputs = ort_session.run(None, {'pixel_values': img})
        feat = outputs[0].mean(axis=1)  # Pool over sequence
        feats.append(feat)
    
    return np.vstack(feats)
```

### Step 3: Add to requirements.txt

```
onnxruntime==1.16.3
onnx==1.15.0
```

### Step 4: Deploy

Upload the .onnx file with your code and redeploy!

## Expected Results

- **Before:** ViT inference ~60s
- **After:** ViT inference ~20-25s
- **Total speedup:** 30-40% faster overall

## Advanced: Quantization (4x memory reduction)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize model to INT8
quantize_dynamic(
    "vit_base.onnx",
    "vit_base_quantized.onnx",
    weight_type=QuantType.QInt8
)
```

Even faster + uses 75% less RAM!

