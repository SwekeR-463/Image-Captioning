# Image-Captioning
kind of implementation of show attend and tell [paper](https://arxiv.org/abs/1502.03044) in pytorch
<br>
the architecture is similar to the paper with efficientnet_b5 encoder, bahdanau attention i.e. soft attention described as per paper and LSTM encoder
<br>

![for_preferred_client](https://github.com/user-attachments/assets/aa27a21e-0d0c-43b5-81a0-01374aa57076)


#### Metrics
Train Acc - 60%<br>
Test Acc - 48%<br>
BLEU Score - 0.0897<br>

### Samples

### Dataset Class
```python
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform=None):
        self.root_dir = root_dir
        self.captions_file = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.captions_file)

    def __getitem__(self, idx):
        img_name = self.captions_file.iloc[idx, 0]
        caption = self.captions_file.iloc[idx, 1]

        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the caption
        caption_tokens = self.tokenizer(caption, padding='max_length', max_length=30, truncation=True, return_tensors="pt")
        caption_tensor = caption_tokens['input_ids'].squeeze()  # Remove extra dimension

        return image, caption_tensor
```

### Attention Part
the attention part basically mentioned in the paper had 2 things - hard and soft.<br>
went with soft attention as hard attention had some interesting stuff like monte carlo sample approximation for gradient descent and that lead to upgradation of parameters
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
The inference part is taken from [this](https://github.com/saurabhaloneai/image-cap/blob/main/src/inference.py)
```python
def predict_caption(image_path, model, tokenizer, max_len=50):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    features = model.encoder(image_tensor)
    
    h, c = model.decoder.init_hidden_state(features)
    
    #Starting the caption with the [CLS] token
    word = torch.tensor([tokenizer.cls_token_id]).to(device)
    embeds = model.decoder.embedding(word)
    
    captions = []
    alphas = []
    
    for _ in range(max_len):
        alpha, context = model.decoder.attention(features, h)
        alphas.append(alpha.cpu().detach().numpy())
        
        lstm_input = torch.cat((embeds.squeeze(1), context), dim=1)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        
        output = model.decoder.fcn(model.decoder.drop(h))
        predicted_word_idx = output.argmax(dim=1)
        
        captions.append(predicted_word_idx.item())
        
        #Break if [SEP] token is generated
        if predicted_word_idx.item() == tokenizer.sep_token_id:
            break
        
        embeds = model.decoder.embedding(predicted_word_idx.unsqueeze(0))
    
    #Converting word indices to words & skipping special tokens
    caption = tokenizer.decode(captions, skip_special_tokens=True)
    return image, caption
```

### BLEU Score
the bleu score measures how similar the generated caption is with the actual caption.
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

### Future Works(maybe)
implementing the hard attention part <br>
shifting to Distributed Data Parallel in [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)<br>
training on Flickr30k<br>
wrapping the model weights in FastAPI and deploying on aws or azure i.e. end2end making<br>
also playing around different hyperparameters for getting the best results<br>
