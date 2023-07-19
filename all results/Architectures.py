import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights 
import torchvision.models as models

class Fusion(nn.Module):
    def __init__(self, model_path, model_path2, TextClassefier, hidden_size=120, output_size=95):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #init embedding model
        self.visual_embedding_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        self.visual_embedding_model = self.visual_embedding_model.to(self.device)
        
        self.visual_embedding_model.fc = nn.Linear(self.visual_embedding_model.fc.in_features, output_size)
        checkpoint = torch.load(model_path)
        self.visual_embedding_model.load_state_dict(checkpoint.pop('model'))
        
        self.text_embedding_model = TextClassefier
        self.text_embedding_model = self.text_embedding_model.to(self.device)
        checkpoint = torch.load(model_path2)
        self.text_embedding_model.load_state_dict(checkpoint.pop('model'))
        # we do not want to overwrite weights for the visual embedding model
        for param in self.visual_embedding_model.parameters():
            param.requires_grad = False
            
        #self.visual_embedding_model.fc = nn.Linear(512, 512)
        
        #init fusion layers
        self.fc1 = nn.Linear(output_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # we have to get a visual embedding first
        
        visual_embedding = self.visual_embedding_model(x['image'].to(self.device, non_blocking=True).type_as(next(self.parameters())))
        text_embedding = self.text_embedding_model(x)
        #x['text_embedding'] = x['text_embedding'].to(dtype=torch.float32)
        #simple fusion via conactention
        x = torch.cat((text_embedding, visual_embedding), dim=1)
        
        # simple MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class TextClassefier(nn.Module):
    def __init__(self, input_size=1536, hidden_size=10000, output_size=95):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        #init fusion layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # we have to get a visual embedding first
        
        
        
        x['text_embedding'] = x['text_embedding'].to(dtype=torch.float32)
        
        
        
        # simple MLP
        x = self.dropout(self.fc1(x['text_embedding'].to(self.device, non_blocking=True)))
        
        x = self.relu(x)
        x = self.dropout(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    
class Transformer_TextClassifier(nn.Module):
    def __init__(self, model_path, input_size=1536, hidden_size=1000, output_size=78, nhead=8, num_layers=4):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Transformer encoder layer
        self.encoder_layer = TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x['text_embedding'] = x['text_embedding'].to(dtype=torch.float32)

        # Transform input to required shape (L, N, E) where L is the sequence length, N is the batch size, E is the feature number
        x = x['text_embedding'].permute(1, 0, 2).to(self.device, non_blocking=True)

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate the output if needed, e.g., taking the mean over sequence length
        x = x.mean(dim=0)

        # Pass through final fully connected layer
        x = self.fc(x)

        return x
    
    
class Fusion_From_Scratch(nn.Module):
    def __init__(self, model_path, input_size=2048, hidden_size=1000, output_size=78):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #init embedding model
        self.visual_embedding_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        self.visual_embedding_model.fc = nn.Linear(512, 512)
        
        #init fusion layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # we have to get a visual embedding first
        
        visual_embedding = self.visual_embedding_model(x['image'].to(self.device, non_blocking=True).type_as(next(self.parameters())))
        
        x['text_embedding'] = x['text_embedding'].to(dtype=torch.float32)
        #simple fusion via conactention
        x = torch.cat((x['text_embedding'].to(self.device, non_blocking=True), visual_embedding), dim=1)
        
        # simple MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x