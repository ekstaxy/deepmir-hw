"""
model.py - Music generation model architectures
"""

from transformers import (
    GPT2Config, 
    GPT2LMHeadModel,
    TransfoXLConfig,
    TransfoXLLMHeadModel
)
import numpy as np
import torch.nn as nn
import torch
import math


class GPT2(nn.Module):    

    def __init__(
        self,
        vocab_size,
        max_seq_len=1024,
        n_embd=512,
        n_layer=12,
        n_head=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):
        super().__init__()
        
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        
        self.model = GPT2LMHeadModel(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TransformerXL(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layer=12,
        n_head=8,
        mem_len=512,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):
        super().__init__()
        
        self.config = TransfoXLConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_head=d_model // n_head,
            d_inner=d_model * 4,
            dropout=0.1,
            dropatt=0.1,
            mem_len=mem_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        
        self.model = TransfoXLLMHeadModel(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None, mems=None):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            mems=mems
        )
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


################################################################################
# Sampling
################################################################################

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)
    
    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# CPWORD Model Components
################################################################################

def network_paras(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CPWordModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(CPWordModel, self).__init__()

        self.n_token = n_token
        self.d_model = 512
        self.n_layer = 12
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [128, 256, 64, 32, 512, 128, 128]
        self.is_training = is_training

        print('>>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type      = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_inner,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layer
        )

        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])        
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, target, loss_mask):
        h, y_type  = self.forward_hidden(x)
        y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity = self.forward_output(h, target)
         
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)
        
        loss_tempo = self.compute_loss(y_tempo, target[..., 0], loss_mask)
        loss_chord = self.compute_loss(y_chord, target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(y_barbeat, target[..., 2], loss_mask)
        loss_type = self.compute_loss(y_type,  target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(y_pitch, target[..., 4], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 6], loss_mask)

        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity

    def forward_hidden(self, x, memory=None, is_training=True):
        emb_tempo =    self.word_emb_tempo(x[..., 0])
        emb_chord =    self.word_emb_chord(x[..., 1])
        emb_barbeat =  self.word_emb_barbeat(x[..., 2])
        emb_type =     self.word_emb_type(x[..., 3])
        emb_pitch =    self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        embs = torch.cat([
            emb_tempo, emb_chord, emb_barbeat, emb_type,
            emb_pitch, emb_duration, emb_velocity
        ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)
    
        if is_training:
            seq_len = pos_emb.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            h = self.transformer_encoder(pos_emb, mask=causal_mask)
            y_type = self.proj_type(h)
            return h, y_type
        else:
            if memory is None:
                memory = {'past_keys': [], 'past_values': []}
            
            pos_emb = pos_emb.squeeze(0)
            h = pos_emb.unsqueeze(0)
            
            for layer in self.transformer_encoder.layers:
                h = layer(h)
            
            h = h.squeeze(0)
            y_type = self.proj_type(h)
            return h, y_type, memory

    def forward_output(self, h, y):
        tf_skip_type = self.word_emb_type(y[..., 3])
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        return y_tempo, y_chord, y_barbeat, y_pitch, y_duration, y_velocity

    def froward_output_sampling(self, h, y_type):
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().to(h.device).unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)
        
        cur_word_tempo =    sampling(y_tempo, t=1.2, p=0.9)
        cur_word_barbeat =  sampling(y_barbeat, t=1.2)
        cur_word_chord =    sampling(y_chord, p=0.99)
        cur_word_pitch =    sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_velocity = sampling(y_velocity, t=5)        

        next_arr = np.array([
            cur_word_tempo, cur_word_chord, cur_word_barbeat,
            cur_word_type, cur_word_pitch, cur_word_duration,
            cur_word_velocity,
        ])        
        return next_arr

    def inference_from_scratch(self, dictionary, device='cuda'):
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([[0, 0, 1, 1, 0, 0, 0]])

        with torch.no_grad():
            final_res = []
            memory = None
            h = None
            
            cnt_bar = 1
            init_t = torch.from_numpy(init).long().to(device)
            print('------ initiate ------')
            for step in range(init.shape[0]):
                print_word_cp(init[step, :])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)

            print('------ generate ------')
            while True:
                next_arr = self.froward_output_sampling(h, y_type)
                final_res.append(next_arr[None, ...])
                print('bar:', cnt_bar, end='  ==')
                print_word_cp(next_arr)

                input_ = torch.from_numpy(next_arr).long().to(device)
                input_  = input_.unsqueeze(0).unsqueeze(0)
                h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)

                if word2event['type'][next_arr[3]] == 'EOS':
                    break
                
                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res


def create_model(model_type, vocab_size, **kwargs):
    if model_type == 'gpt2':
        return GPT2(vocab_size=vocab_size, **kwargs)
    elif model_type == 'transformer-xl':
        return TransformerXL(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cpword':
        return CPWordModel(n_token=vocab_size, is_training=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing model creation...")
    
    n_token = [128, 256, 64, 32, 512, 128, 128]
    model = CPWordModel(n_token=n_token, is_training=True)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CPWord Model created: {total_params:,} parameters")
    
    batch_size = 4
    seq_len = 512
    dummy_input = torch.randint(0, 50, (batch_size, seq_len, 7))
    dummy_target = torch.randint(0, 50, (batch_size, seq_len, 7))
    dummy_mask = torch.ones(batch_size, seq_len)
    
    losses = model.train_step(dummy_input, dummy_target, dummy_mask)
    print(f"Test forward pass successful!")
    print(f"Loss values: {[f'{l.item():.4f}' for l in losses]}")