import hashlib
import torch
import numpy as np
import random
import hashlib

def get_new_delta(initial_weights, delta, compression_function = lambda x:x):
    new_delta = initial_weights - (initial_weights + delta) #initial_weights - reconstructed_weights
    return compression_function(new_delta)

def reconstruct_delta(initial_weights, delta, decompression_function = lambda x:x):
    reconstructed_weight = initial_weights + delta
    return decompression_function(reconstructed_weight)

def randomize_weights(model):
    with torch.no_grad():  # Ensure no gradient computation during weight randomization
        for name, param in model.state_dict().items():
            if "weight" in name:  # If the tensor is a weight
                # Randomly initialize the weight using the same size
                param.copy_(torch.randn_like(param))
            elif "bias" in name:  # If the tensor is a bias
                # Randomly initialize the bias using the same size
                param.copy_(torch.randn_like(param))



class TrainingModel:

    def __init__(self, ModelToUse=None, model=None):
        if model is not None:
            self._model = model
        else:
            self._model = ModelToUse()
        
        # Storing the initial weights
        self.initial_weights = {name: tensor.clone() for name, tensor in self._model.state_dict().items()}

    @property
    def model(self):
        return self._model
    
    @staticmethod
    def hash_model_weights(model):
        state_dict_bytes = b''.join(map(torch.Tensor.tobytes, model.state_dict().values()))
        return hashlib.sha256(state_dict_bytes).hexdigest()

    @property
    def concatenated_weights(self):
        return torch.cat([tensor.flatten() for tensor in self._model.state_dict().values()])

    @property
    def concatenated_initial_weights(self):
        """
        Getter for concatenated initial weights.
        Returns a single tensor with all the model's initial weights concatenated.
        """
        return torch.cat([tensor.flatten() for tensor in self.initial_weights.values()])

    @concatenated_weights.setter
    def concatenated_weights(self, concatenated_tensor):
        import pdb;pdb.set_trace()
        assert concatenated_tensor.numel() == sum(tensor.numel() for tensor in self._model.state_dict().values())
        tensor_iter = iter(concatenated_tensor)
        
        state_dict = {}
        for name, tensor in self._model.state_dict().items():
            extracted_elements = [tensor_iter.__next__() for _ in range(tensor.numel())]
            state_dict[name] = torch.tensor(extracted_elements, dtype=tensor.dtype).reshape_as(tensor)
        
        self._model.load_state_dict(state_dict)

    def apply_delta(self, delta_tensor):
        """
        Update the model weights by adding delta to the initial weights.

        :param delta_tensor: Tensor - The concatenated delta weights tensor.
        """
        import pdb;pdb.set_trace()
        assert delta_tensor.numel() == sum(tensor.numel() for tensor in self.initial_weights.values())
        tensor_iter = iter(delta_tensor)

        # Preparing a dictionary to hold the modified state dict
        modified_state_dict = {}

        for name, tensor in self.initial_weights.items():
            # Extracting elements for each tensor from the iterator
            extracted_elements = [tensor_iter.__next__() for _ in range(tensor.numel())]
            delta = torch.tensor(extracted_elements, dtype=tensor.dtype).reshape_as(tensor)
            
            # Adding the delta to the initial weights
            modified_state_dict[name] = tensor + delta
        
        # Loading the modified state dict into the model
        self._model.load_state_dict(modified_state_dict)
