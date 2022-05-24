import torch
import torch.nn as nn

class Distance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.output2 = nn.Linear(self.hidden_size, 1)
    
    def forward(self, sentence1, sentence2, labels, mean_dis, std_dis):
        sentence = torch.cat([sentence1, sentence2, torch.abs(sentence1-sentence2)], dim=1)
        outputs = torch.relu(self.output1(sentence))
        outputs = self.output2(outputs)
        outputs = outputs.squeeze(1)
        outputs = outputs * std_dis + mean_dis
        losses = torch.sum((outputs-labels) ** 2)
        loss = losses / sentence1.size(0)
        return loss