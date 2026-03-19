import onnx, onnxruntime as ort, numpy as np
m = onnx.load('treedetect.onnx')
print('graph input', [(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('graph output', [(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
s = ort.InferenceSession('treedetect.onnx')
inp = np.random.randn(1,3,640,640).astype(np.float32)
out = s.run(None,{s.get_inputs()[0].name: inp})
print('output count', len(out))
for i,o in enumerate(out):
    print(i, o.shape, o.dtype, float(o.min()), float(o.max()))
