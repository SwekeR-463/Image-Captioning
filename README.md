# Image-Captioning
kind of implementation of show attend and tell [paper](https://arxiv.org/abs/1502.03044)
<br>
the architecture is similar to the paper with resnet encoder, bahdanau attention i.e. soft attention described as per paper and LSTM encoder
<br>

#### Metrics
Train Acc - 60%<br>
Test Acc - 48%<br>
BLEU Score - 0.0897<br>

### Dataset Class
```
```

### Attention Part
```python
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
```python
from nltk.translate.bleu_score import corpus_bleu

def calculate_bleu(predicted_captions, ground_truth_captions):

    predicted_captions = [caption.split() for caption in predicted_captions]
    ground_truth_captions = [[caption.split()] for caption in ground_truth_captions]  
    
    bleu_score = corpus_bleu(ground_truth_captions, predicted_captions)
    return bleu_score

def evaluate_model(model, test_loader, tokenizer):
    model.eval()
    predicted_captions = []
    ground_truth_captions = []
    
    with torch.no_grad():
        for image, captions in test_loader:
            image = image.to(device)
            captions = captions.to(device)
            
            outputs, _ = model(image, captions)
            
            _, predicted = outputs.max(2)
            predicted = predicted.cpu().numpy()

            for idx in range(predicted.shape[0]):
                predicted_caption = tokenizer.decode(predicted[idx], skip_special_tokens=True)
                predicted_captions.append(predicted_caption)
                
                ground_truth_caption = tokenizer.decode(captions[idx, 1:], skip_special_tokens=True)
                ground_truth_captions.append(ground_truth_caption)

    #Calculate BLEU score
    bleu_score = calculate_bleu(predicted_captions, ground_truth_captions)
    print(f"BLEU Score: {bleu_score:.4f}")
    return bleu_score
```
