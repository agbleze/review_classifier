#%%
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

token_ids = [847, 23, 112, 592, 9, 847, 23, 112, 592, 
             847, 23, 112, 23, 112, 592, 9, 
             847, 23, 112, 592, 847, 23, 112,
             23, 112, 592, 9, 847, 23,
             847, 23, 112, 592, 847, 23, 112,
             23, 112, 592, 9, 847, 23
             ]
sequence = np.array([token_ids], dtype=np.int64)
model_path = "/home/lin/codebase/review_classifier/model_store/model_state.onnx"
op_model_path = "/home/lin/codebase/review_classifier/model_store/optimized_model.onnx"
session = ort.InferenceSession(op_model_path)
#input_tensor = sequence.reshape(1, 1, -1)
inputs = {session.get_inputs()[0].name: sequence}

outputs = session.run(None, inputs)
outputs
#%%

len(token_ids)

#%%

logits = softmax(outputs[0], axis=1)
logits
#%%
np.argmax(logits)


#%%

softmax([10, 10])

# %%
for i in session.get_inputs():
    print("Name:", i.name)
    print("Shape:", i.shape)
    print("Type:", i.type)

# %%
session._model_meta

# %%
