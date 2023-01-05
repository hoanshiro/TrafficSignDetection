mo --input_model "onnx_model/FasterRCNN.onnx" --input_shape "[1,3, 626, 1622]" \
 --mean_values="[123.675, 116.28 , 103.53]" --scale_values="[58.395, 57.12 , 57.375]" \
  --data_type FP16 --output_dir "model/FasterRCNN"
