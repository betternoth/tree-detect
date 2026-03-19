import onnxruntime as ort, numpy as np
import os
print('cwd', os.getcwd())
model_path = 'treedetect.onnx'
print('exists', os.path.exists(model_path), model_path)

s = ort.InferenceSession(model_path)
print('input', s.get_inputs()[0].name, s.get_inputs()[0].shape, s.get_inputs()[0].type)
print('output', s.get_outputs()[0].name, s.get_outputs()[0].shape, s.get_outputs()[0].type)
input_tensor = np.random.rand(1,3,640,640).astype(np.float32)
out = s.run(None, {s.get_inputs()[0].name: input_tensor})
print('out shapes', [o.shape for o in out])
out0 = out[0]
print('out0 stats', out0.min(), out0.max(), out0.mean())
print('sample first 20', out0.flatten()[:20])
print('sample first 5 rows', out0.reshape(out0.shape[0], out0.shape[1], out0.shape[2])[:,:,:5])
