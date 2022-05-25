import torch
import torch.nn as nn

class Distance(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.hidden_size = config.hidden_size
        self.output1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.output2 = nn.Linear(self.hidden_size, 1)
    
    def forward(self, text1_ids, text1_pads, text2_ids, text2_pads):
        outputs1 = self.model(input_ids=text1_ids, attention_mask=text1_pads)
        encoded1 = outputs1[0][:, 0]
        outputs2 = self.model(input_ids=text2_ids, attention_mask=text2_pads)
        encoded2 = outputs2[0][:, 0]
        encoded = torch.cat([encoded1, encoded2], dim=1)
        outputs = self.output1(encoded)
        outputs = self.output2(outputs)
        return outputs.squeeze(1)
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)