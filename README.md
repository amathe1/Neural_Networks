# Neural_Networks

**Basics of Neural Networks and some important points to remember :**

**1) torch.nn** is a module that contains many classes and functions, such as:
- nn.Module (base class for all neural networks)
- nn.Linear (use for Artificial Neural Networks)
- nn.ReLU
- nn.Conv2d (use when i/p data is an image)
- nn.CrossEntropyLoss
- nn.Sequential
- nn.RNN (use for sequence to sequence data)
- nn.LSTM (use for sequence to sequence data)

**2)** **torch/                (package)
    └── nn/              (submodule)
         ├── module.py    → contains class Module
         ├── linear.py    → contains class Linear
         ├── activation.py**

**3)** We are also going to get Activation functions, Loss & Backward propagation(nn.train(), nn.eval()) from nn submodule. 

**4)** In practice, Artificial Neural Network refers to a fully connected feed-forward neural network built using Linear layers and activations.

**5)** There is no fundamental difference between ANN and a neural network — ANN is simply a subset of neural networks, 
   while CNNs, RNNs, and Transformers are specialized architectures.

**6)** **self represents the current object (instance) of the class**.
    model = SimpleNet() ; model is an object
    Inside the class, Python calls that object self
    So, self  ⟷  model
    
    Why self is REQUIRED here ?
      Storing layers inside the object
      self.fc1 = nn.Linear(3, 5)
      self.fc2 = nn.Linear(5, 2)
      This means “Attach fc1 and fc2 to this model object”
      
      Without self : fc1 = nn.Linear(3, 5) ; fc2 = nn.Linear(5, 2)
        - fc1 = nn.Linear(3, 5)   # ❌ local variable
        - fc1 would exist only inside __init__()
        - It would be destroyed once __init__ finishes
        - PyTorch would not know this layer belongs to the model

**7)** **Why self is used again in forward() ?**
    -  x = self.fc1(x)
    -  x = self.fc2(x)
    -  because fc1 and fc2 belong to the object, you must access them via object(self)
    -  This is exactly same as :
            model.fc1(x)
            model.fc2(x)

**One liner answer for self :**
self is used to bind layers and variables to the model instance so that PyTorch can track them as part of the network, 
register their parameters, and make them accessible in the forward pass.

****8)** Internally, PyTorch does something like this:**
    def __call__(self, *args, **kwargs):
    # 1. Pre-forward hooks
    # 2. Call forward()
    output = self.forward(*args, **kwargs)
    # 3. Post-forward hooks
    return output
