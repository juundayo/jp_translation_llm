# ----------------------------------------------------------------------------#

'''
Dataset:
English-Japanese version in http://www.manythings.org/anki/ 
'''

# ----------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import Counter
import MeCab
import unidic_lite
import math
import copy
import re

# ----------------------------------------------------------------------------#

'''
Dataset format: 
English sentence     Japanese Sentence     Source
Symbols are used as separators to extract the sentences.
'''
DATASET_PATH = "/home/ml3/Desktop/Thesis/.venv/jpn.txt"

'''Dataset variables.'''
SRC_VOCAB_SIZE = 10000
TGT_VOCAB_SIZE = 10000
MAX_SQL_LENGTH = 50
VOCAB_SIZE = 10000
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mecab = MeCab.Tagger("-d " + unidic_lite.DICDIR + " -Owakati")
TRAIN = False
EPOCHS = 25
BATCH_SIZE = 32
D_MODEL = 256
NUM_HEADS = 2
NUM_LAYERS = 4
D_FF = 512

# ----------------------------------------------------------------------------#

'''Data preparation: Tokenization, Vocabulary and sentence encoding.'''

def tokenize(sentence):
    return re.findall(r"\b\w+\b", sentence.lower())

def tokenize_japanese(sentence):    
    if sentence.strip() == "":
        return []
    result = mecab.parse(sentence)
    if result is None:
        return []
    
    return result.strip().split()

def build_vocab(sentences, lang, max_vocab_size):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenize_japanese(sentence) if lang == 'jpn' else tokenize(sentence)
        counter.update(tokens)

    # Getting the most common words. -4 used for special characters.
    most_common = counter.most_common(max_vocab_size - 4)
    idx2word = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + [word for word, _ in most_common]
    word2idx = {word: idx for idx, word in enumerate(idx2word)}

    if lang == 'jpn':
        for sentence in sentences[:5]:
            print("Original JP:", sentence)
            print("Tokens JP:", tokenize_japanese(sentence))

    return idx2word, word2idx

def encode_japanese(sentence, word2idx, max_len=MAX_SQL_LENGTH):
    tokens = tokenize_japanese(sentence)
    if not tokens:
        tokens = ['<UNK>']
    
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    indices = [word2idx['<SOS>']] + indices + [word2idx['<EOS>']]
    
    if len(indices) < max_len:
        indices += [word2idx['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return indices

def encode_sentence(sentence, word2idx, max_len=MAX_SQL_LENGTH):
    tokens = tokenize(sentence)

    if not tokens:
        tokens = ['<UNK>']

    indices = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]
    indices = [word2idx["<SOS>"]] + indices + [word2idx["<EOS>"]]

    if len(indices) < max_len:
        indices += [word2idx["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    return indices

# ----------------------------------------------------------------------------#

class TranslationDataset(data.Dataset):
    def __init__(self, filepath, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE):
        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()

        eng_sentences, jpn_sentences = [], []
        for line in lines:
            if '\t' not in line: # Tab seperators.
                continue

            eng, jpn = line.strip().split('\t')[:2]
            eng_sentences.append(eng)
            jpn_sentences.append(jpn)

        self.src_sentences = eng_sentences
        self.tgt_sentences = jpn_sentences

        self.idx2src, self.src_word2idx = build_vocab(eng_sentences, 
                                                      lang='eng', 
                                                      max_vocab_size=SRC_VOCAB_SIZE)
        self.idx2tgt, self.tgt_word2idx = build_vocab(jpn_sentences, 
                                                      lang='jpn', 
                                                      max_vocab_size=TGT_VOCAB_SIZE)

        self.pairs = [
            (encode_sentence(src, self.src_word2idx),
             encode_japanese(tgt, self.tgt_word2idx))
             for src, tgt in zip(eng_sentences, jpn_sentences)
        ]

    def __len__(self):
            return len(self.pairs)
        
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src), torch.tensor(tgt)
    
# ----------------------------------------------------------------------------#

class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, 
                      self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, 
                                                   self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output

# ----------------------------------------------------------------------------#

class PositionWiseFeedForward(nn.Module):
    '''
    Enables the model to consider the position of 
    input elements while making predictions.
    '''
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
# ----------------------------------------------------------------------------#

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
# ----------------------------------------------------------------------------#

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
# ----------------------------------------------------------------------------#

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads)
        self.cross_attn = Attention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
# ----------------------------------------------------------------------------#

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                 num_layers, d_ff, max_seq_length):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(
                d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(
                d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(
            torch.ones(1, seq_length, seq_length, device=src.device), 
            diagonal=1)).bool()

        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(
                                    self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(
                                    self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
# ----------------------------------------------------------------------------#

def get_loader(DATASET_PATH):
    dataset = TranslationDataset(DATASET_PATH, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    return data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                            shuffle=True), dataset

def translate(model, src_sentence, src_word2idx, tgt_idx2word, max_length=50):
    model.eval()
    src_encoded = encode_sentence(src_sentence, src_word2idx)
    src_tensor = torch.tensor(src_encoded).unsqueeze(0).to(DEVICE)

    generated = [SOS_IDX]
    for _ in range(max_length):
        tgt_tensor = torch.tensor([generated], device=DEVICE)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        logits = output[0, -1]
        logits[UNK_IDX] = -float('inf')  # Block UNK.
        logits[PAD_IDX] = -float('inf')  # Block PAD.
        
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
        if next_token == EOS_IDX:
            break

    # Converting to text and filtering special tokens.
    return ' '.join(tgt_idx2word[idx] for idx in generated 
                   if idx not in [SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX])

# ----------------------------------------------------------------------------#

if __name__ == '__main__':    
    dataloader, dataset = get_loader(DATASET_PATH)
    print("Loaded data successfully.")

    print("\n=== Data Verification ===")
    print("Sample training pair:")
    print("EN:", dataset.src_sentences[1000])
    print("JP:", dataset.tgt_sentences[1000])
    print("Encoded JP:", [dataset.idx2tgt[i] for i in encode_japanese(dataset.tgt_sentences[1000], 
                                                                      dataset.tgt_word2idx)])

    model = Transformer(
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, 
        NUM_HEADS, NUM_LAYERS, D_FF, MAX_SQL_LENGTH).to(DEVICE)
    
    if TRAIN:
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            
            for src_data, tgt_data in dataloader:
                src_data, tgt_data = src_data.to(DEVICE), tgt_data.to(DEVICE)
                
                optimizer.zero_grad()
                output = model(src_data, tgt_data[:, :-1])
                
                loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), 
                                tgt_data[:, 1:].contiguous().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "llm.pth")
    else:
        model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL,
                    NUM_HEADS, NUM_LAYERS, D_FF, MAX_SQL_LENGTH).to(DEVICE)
        model.load_state_dict(torch.load("llm.pth"))
        model.eval()

        print("\n=== Model Output Verification ===")
        print()
        
        word = "It will be a sunny day tomorrow."
        print("Input: ", word)
        translated = translate(model, word, dataset.src_word2idx, dataset.idx2tgt)
        print("Translation:", translated)
        
        test_src = "It is old.?"
        print("Input:", test_src)
        src_encoded = encode_sentence(test_src, dataset.src_word2idx)
        src_tensor = torch.tensor(src_encoded).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            test_output = model(src_tensor, torch.tensor([[SOS_IDX]], device=DEVICE))
        
        topk_values, topk_indices = torch.softmax(test_output[0,0], dim=-1).topk(5)
        print("First prediction distribution:")
        for val, idx in zip(topk_values, topk_indices):
            print(f"{dataset.idx2tgt[idx.item()]}: {val.item():.4f}")

