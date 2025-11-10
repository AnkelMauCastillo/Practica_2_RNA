# src/mlp_gpu.py
import torch
import torch.nn as nn
import numpy as np

class MLP_GPU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, inicializacion='xavier'):
        super(MLP_GPU, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Usando dispositivo: {self.device}")
        
        # Capa oculta y capa de salida
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos
        self._inicializar_pesos(inicializacion)
        
        # Mover modelo a GPU
        self.to(self.device)
        
    def _inicializar_pesos(self, tipo):
        """Aplica diferentes estrategias de inicialización"""
        if tipo == 'xavier':
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
            print(" Inicialización Xavier aplicada")
        elif tipo == 'normal':
            nn.init.normal_(self.hidden.weight, mean=0, std=0.01)
            nn.init.normal_(self.output.weight, mean=0, std=0.01)
            print(" Inicialización Normal aplicada")
        
        # Inicializar biases a cero
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        """Forward propagation"""
        # Asegurar que x está en el dispositivo correcto
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        elif isinstance(x, torch.Tensor):
            x = x.to(self.device)
            
        x = self.hidden(x)
        x = self.sigmoid(x)  # Sigmoid en capa oculta
        x = self.output(x)
        x = self.sigmoid(x)  # Sigmoid en capa de salida
        return x
    
    def predict_proba(self, x):
        """Predicciones en probabilidades"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            probas = self.forward(x)
            return probas.cpu().numpy()
    
    def predict(self, x, threshold=0.5):
        """Predicciones binarias"""
        probas = self.predict_proba(x)
        return (probas > threshold).astype(int)
    
    def get_parameters_count(self):
        """Cuenta el número total de parámetros"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params