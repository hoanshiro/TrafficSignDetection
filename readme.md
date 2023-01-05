Preparation:
1. Crate virtual environment  ``conda create -n venv python=3.8``
2. Install requirements ``!pip install -r requirements``
3. Convert Torch to ONNX Model [Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
4. Convert ONNX Model to OpenVINO IR Format``$bash onnx2ir.sh``
5. Inference ``python inference.py``

Have fun!