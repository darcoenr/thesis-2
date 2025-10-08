import torch
import torch.nn as nn

from transformers import BertModel

class ClassificationHead(nn.Module):
    def __init__(self, input_size, number_of_layers, layer_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for _ in range(number_of_layers - 1):
            layers.append([nn.Linear(prev_size, layer_size), nn.Tanh()])
            prev_size = layer_size
        layers = sum(layers, [])
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        return logits
  
class AverageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embedding, special_tokens_mask):
        # Set the embedding relative to special tokens (i.e. CLS, SEP, PAD) to zero
        embedding[special_tokens_mask != 0] = 0
        # Sum the embedding along the second dimension ('sequence length')
        embedding_sum = torch.sum(embedding, dim=1)
        # Get the number of 'real' tokens
        seq_length = torch.sum(special_tokens_mask == 0, dim=1).unsqueeze(1)
        # Compute the mean
        mean = embedding_sum / seq_length
        return mean

class igbert_wrapper(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        # Download the specified BERT model.
        self.bert_model = BertModel.from_pretrained(checkpoint, add_pooling_layer=False)
        # Freeze the model (disable gradient computation)
        for p in self.bert_model.parameters():
            p.requires_grad = False
    def forward(self, input_ids, attention_mask, special_tokens_mask):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return bert_output.last_hidden_state

class ClassificationFromAveraging(nn.Module):
    def __init__(self, checkpoint, number_of_layers, layer_size):
        super().__init__()
        if checkpoint == 'Exscientia/IgBert':
            self.encoder = igbert_wrapper(checkpoint)
        hidden_size = 1024
        self.mean = AverageEmbedding()
        self.classification_head = ClassificationHead(hidden_size, number_of_layers, layer_size)
    def forward(self, input_ids, attention_mask, special_tokens_mask):
        embeddings = self.encoder(input_ids, attention_mask, special_tokens_mask)
        mean = self.mean(embeddings, special_tokens_mask)
        logits = self.classification_head(mean)
        return logits