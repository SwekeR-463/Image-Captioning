# Image-Captioning
kind of implementation of show attend and tell [paper](https://arxiv.org/abs/1502.03044)
<br>
the architecture is similar to the paper with resnet encoder, bahdanau attention i.e. soft attention described as per paper and LSTM encoder
<br>

### Dataset Class
```
```

### Attention Part
```
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        #encoder hidden states hj
        self.Wh = nn.Linear(enc_hid_dim, attn_dim)
        #decoder previous hidden state si-1
        self.Ws = nn.Linear(dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, enc_out, dec_hid):
        enc_features = self.Wh(enc_out)
        dec_features = self.Ws(dec_hid).unsqueeze(1)
        
        scores = torch.tanh(enc_features + dec_features)
        #eij
        alignment_scores = self.v(scores).squeeze(-1) 
        
        #alphaij
        attention_weights = F.softmax(alignment_scores, dim=1)
        
        #cij
        context = torch.bmm(attention_weights.unsqueeze(1), enc_out).squeeze(1)
        
        return context, attention_weights
    
```

### Inference
```
```

### BLEU Score
```
```
