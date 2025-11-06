"""
model.py - Music generation model architectures
FIXED: CPWordModel now matches MidiTok CPWord tokenization output
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
    """
    FIXED CPWord Model for MidiTok CPWord tokenization
    
    Token structure (8 vocabularies):
        Index 0: Family (Type) - note vs non-note events
        Index 1: Bar - bar markers and positions  
        Index 2: Pitch - MIDI pitch values
        Index 3: Velocity - note velocities
        Index 4: Duration - note durations
        Index 5: Chord - chord tokens
        Index 6: Rest - rest/time-shift tokens
        Index 7: Tempo - tempo values
    
    Vocabulary sizes: [6, 38, 156, 128, 101, 173, 36, 37]
    """
    
    def __init__(self, n_token=None, tokenizer=None, is_training=True):
        super(CPWordModel, self).__init__()

        # Get vocabulary sizes from tokenizer or use provided values
        if tokenizer is not None:
            self.n_token = [len(tokenizer.vocab[i]) for i in range(len(tokenizer.vocab))]
            print(f"✓ Loaded vocab sizes from tokenizer: {self.n_token}")
        elif n_token is not None:
            self.n_token = n_token
            print(f"✓ Using provided vocab sizes: {self.n_token}")
        else:
            raise ValueError("Must provide either n_token or tokenizer")
        
        # Verify we have 8 vocabularies
        if len(self.n_token) != 8:
            raise ValueError(f"Expected 8 vocabularies, got {len(self.n_token)}")
        
        # Model hyperparameters
        self.d_model = 512
        self.n_layer = 12
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.is_training = is_training
        
        # Embedding sizes for each vocabulary
        # Adjusted based on vocabulary sizes: [6, 38, 156, 128, 101, 173, 36, 37]
        self.emb_sizes = [
            32,   # Family (small vocab: 6 tokens)
            64,   # Bar (small vocab: 38 tokens)
            256,  # Pitch (large vocab: 156 tokens) 
            128,  # Velocity (medium vocab: 128 tokens)
            128,  # Duration (medium vocab: 101 tokens)
            256,  # Chord (large vocab: 173 tokens)
            64,   # Rest (small vocab: 36 tokens)
            64,   # Tempo (small vocab: 37 tokens)
        ]
        
        print(f"Model configuration:")
        print(f"  Vocabularies: {len(self.n_token)}")
        print(f"  Vocab sizes: {self.n_token}")
        print(f"  Embedding sizes: {self.emb_sizes}")
        print(f"  Total embedding dim: {sum(self.emb_sizes)} -> {self.d_model}")
        
        # Create embeddings for each vocabulary
        # Order: family/bar/pitch/velocity/duration/chord/rest/tempo
        self.word_emb_family   = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_bar      = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_pitch    = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_velocity = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_duration = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_chord    = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_rest     = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.word_emb_tempo    = Embeddings(self.n_token[7], self.emb_sizes[7])
        
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout)

        # Linear projection from concatenated embeddings to model dimension
        self.in_linear = nn.Linear(sum(self.emb_sizes), self.d_model)

        # Transformer encoder
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

        # Skip connection with family embedding for output projection
        self.project_concat_family = nn.Linear(self.d_model + self.emb_sizes[0], self.d_model)

        # Output projection heads for each vocabulary
        self.proj_family   = nn.Linear(self.d_model, self.n_token[0])
        self.proj_bar      = nn.Linear(self.d_model, self.n_token[1])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[2])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[5])
        self.proj_rest     = nn.Linear(self.d_model, self.n_token[6])
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[7])
        
        print(f"✓ CPWordModel initialized successfully")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")

    def compute_loss(self, predict, target, loss_mask):
        """Compute masked loss for a single vocabulary"""
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, target, loss_mask):
        """
        Training step for CPWord model
        
        Args:
            x: [batch, seq_len, 8] - input tokens
            target: [batch, seq_len, 8] - target tokens (shifted by 1)
            loss_mask: [batch, seq_len] - mask for loss computation
        
        Returns:
            Tuple of 8 loss values (one per vocabulary)
        """
        # Get hidden states and family predictions
        h, y_family = self.forward_hidden(x)
        
        # Get predictions for other vocabularies
        y_bar, y_pitch, y_velocity, y_duration, y_chord, y_rest, y_tempo = \
            self.forward_output(h, target)
        
        # Permute predictions to [batch, vocab_size, seq_len] for CrossEntropyLoss
        y_family   = y_family.permute(0, 2, 1)
        y_bar      = y_bar.permute(0, 2, 1)
        y_pitch    = y_pitch.permute(0, 2, 1)
        y_velocity = y_velocity.permute(0, 2, 1)
        y_duration = y_duration.permute(0, 2, 1)
        y_chord    = y_chord.permute(0, 2, 1)
        y_rest     = y_rest.permute(0, 2, 1)
        y_tempo    = y_tempo.permute(0, 2, 1)
        
        # Compute losses for each vocabulary
        loss_family   = self.compute_loss(y_family,   target[..., 0], loss_mask)
        loss_bar      = self.compute_loss(y_bar,      target[..., 1], loss_mask)
        loss_pitch    = self.compute_loss(y_pitch,    target[..., 2], loss_mask)
        loss_velocity = self.compute_loss(y_velocity, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(y_duration, target[..., 4], loss_mask)
        loss_chord    = self.compute_loss(y_chord,    target[..., 5], loss_mask)
        loss_rest     = self.compute_loss(y_rest,     target[..., 6], loss_mask)
        loss_tempo    = self.compute_loss(y_tempo,    target[..., 7], loss_mask)

        return loss_family, loss_bar, loss_pitch, loss_velocity, loss_duration, \
               loss_chord, loss_rest, loss_tempo

    def forward_hidden(self, x, memory=None, is_training=True):
        """
        Forward pass to get hidden states and family predictions
        
        Args:
            x: [batch, seq_len, 8] - input compound tokens
            memory: (optional) memory for inference
            is_training: whether in training mode
        
        Returns:
            h: [batch, seq_len, d_model] - hidden states
            y_family: [batch, seq_len, n_family] - family predictions
            (memory): memory state if not training
        """
        # Extract embeddings for each vocabulary
        # Order: family/bar/pitch/velocity/duration/chord/rest/tempo
        emb_family   = self.word_emb_family(x[..., 0])
        emb_bar      = self.word_emb_bar(x[..., 1])
        emb_pitch    = self.word_emb_pitch(x[..., 2])
        emb_velocity = self.word_emb_velocity(x[..., 3])
        emb_duration = self.word_emb_duration(x[..., 4])
        emb_chord    = self.word_emb_chord(x[..., 5])
        emb_rest     = self.word_emb_rest(x[..., 6])
        emb_tempo    = self.word_emb_tempo(x[..., 7])

        # Concatenate all embeddings
        embs = torch.cat([
            emb_family, emb_bar, emb_pitch, emb_velocity,
            emb_duration, emb_chord, emb_rest, emb_tempo
        ], dim=-1)

        # Project to model dimension and add positional encoding
        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)
    
        if is_training:
            # Create causal mask for autoregressive training
            seq_len = pos_emb.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            
            # Pass through transformer
            h = self.transformer_encoder(pos_emb, mask=causal_mask)
            
            # Predict family tokens directly
            y_family = self.proj_family(h)
            
            return h, y_family
        else:
            # Inference mode (for generation)
            if memory is None:
                memory = {'past_keys': [], 'past_values': []}
            
            pos_emb = pos_emb.squeeze(0)
            h = pos_emb.unsqueeze(0)
            
            for layer in self.transformer_encoder.layers:
                h = layer(h)
            
            h = h.squeeze(0)
            y_family = self.proj_family(h)
            
            return h, y_family, memory

    def forward_output(self, h, y):
        """
        Generate predictions for all vocabularies except family
        
        Args:
            h: [batch, seq_len, d_model] - hidden states from forward_hidden
            y: [batch, seq_len, 8] - target tokens (used for skip connection)
        
        Returns:
            Tuple of predictions for: bar, pitch, velocity, duration, chord, rest, tempo
        """
        # Skip connection: concatenate hidden state with family embedding
        tf_skip_family = self.word_emb_family(y[..., 0])
        y_concat_family = torch.cat([h, tf_skip_family], dim=-1)
        y_ = self.project_concat_family(y_concat_family)

        # Project to each vocabulary
        y_bar      = self.proj_bar(y_)
        y_pitch    = self.proj_pitch(y_)
        y_velocity = self.proj_velocity(y_)
        y_duration = self.proj_duration(y_)
        y_chord    = self.proj_chord(y_)
        y_rest     = self.proj_rest(y_)
        y_tempo    = self.proj_tempo(y_)

        return y_bar, y_pitch, y_velocity, y_duration, y_chord, y_rest, y_tempo

    def forward_output_sampling(self, h, y_family):
        """
        Sample next token during generation
        
        Args:
            h: [d_model] - hidden state for current position
            y_family: [n_family] - family logits
        
        Returns:
            next_arr: [8] - sampled token for each vocabulary
        """
        # Sample family token
        y_family_logit = y_family[0, :]
        cur_word_family = sampling(y_family_logit, p=0.90)

        # Create skip connection with sampled family token
        family_word_t = torch.from_numpy(
            np.array([cur_word_family])).long().to(h.device).unsqueeze(0)

        tf_skip_family = self.word_emb_family(family_word_t).squeeze(0)
        y_concat_family = torch.cat([h, tf_skip_family], dim=-1)
        y_ = self.project_concat_family(y_concat_family)

        # Get logits for all vocabularies
        y_bar      = self.proj_bar(y_)
        y_pitch    = self.proj_pitch(y_)
        y_velocity = self.proj_velocity(y_)
        y_duration = self.proj_duration(y_)
        y_chord    = self.proj_chord(y_)
        y_rest     = self.proj_rest(y_)
        y_tempo    = self.proj_tempo(y_)
        
        # Sample from each vocabulary with different temperatures/nucleus
        cur_word_bar      = sampling(y_bar, t=1.2)
        cur_word_pitch    = sampling(y_pitch, p=0.9)
        cur_word_velocity = sampling(y_velocity, t=5)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_chord    = sampling(y_chord, p=0.99)
        cur_word_rest     = sampling(y_rest, t=1.2)
        cur_word_tempo    = sampling(y_tempo, t=1.2, p=0.9)

        # Combine into compound token
        # Order: family/bar/pitch/velocity/duration/chord/rest/tempo
        next_arr = np.array([
            cur_word_family, cur_word_bar, cur_word_pitch, cur_word_velocity,
            cur_word_duration, cur_word_chord, cur_word_rest, cur_word_tempo
        ])
        
        return next_arr

    def inference_from_scratch(self, dictionary, device='cuda'):
        """
        Generate music from scratch
        
        Args:
            dictionary: (event2word, word2event) mappings
            device: device to run on
        
        Returns:
            final_res: [seq_len, 8] - generated token sequence
        """
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]
            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        # Initial token: [0, 0, 0, 0, 0, 0, 0, 0] or appropriate BOS
        init = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

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
                h, y_family, memory = self.forward_hidden(input_, memory, is_training=False)

            print('------ generate ------')
            max_len = 2000  # Maximum generation length
            for gen_step in range(max_len):
                next_arr = self.forward_output_sampling(h, y_family)
                final_res.append(next_arr[None, ...])
                
                print('bar:', cnt_bar, end='  ==')
                print_word_cp(next_arr)

                input_ = torch.from_numpy(next_arr).long().to(device)
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_family, memory = self.forward_hidden(input_, memory, is_training=False)

                # Check for EOS or bar count
                if word2event['family'][next_arr[0]] == 'EOS':
                    break
                
                if 'Bar' in str(word2event['bar'][next_arr[1]]):
                    cnt_bar += 1
                
                if cnt_bar > 32:  # Limit generation length
                    break

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res


def create_model(model_type, vocab_size=None, tokenizer=None, **kwargs):
    """
    Create a model based on type
    
    Args:
        model_type: 'gpt2', 'transformer-xl', or 'cpword'
        vocab_size: vocabulary size (for GPT2/TransformerXL) or list of sizes (for CPWord)
        tokenizer: tokenizer object (for CPWord, to auto-extract vocab sizes)
        **kwargs: additional model arguments
    """
    if model_type == 'gpt2':
        return GPT2(vocab_size=vocab_size, **kwargs)
    elif model_type == 'transformer-xl':
        return TransformerXL(vocab_size=vocab_size, **kwargs)
    elif model_type == 'cpword':
        return CPWordModel(n_token=vocab_size, tokenizer=tokenizer, is_training=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing CPWord model creation...")
    
    # Test with explicit vocab sizes (your tokenizer output)
    n_token = [6, 38, 156, 128, 101, 173, 36, 37]
    print(f"\nCreating model with vocab sizes: {n_token}")
    
    model = CPWordModel(n_token=n_token, is_training=True)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ CPWord Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 128
    
    # Create dummy data with correct vocab ranges
    dummy_input = torch.zeros((batch_size, seq_len, 8), dtype=torch.long)
    dummy_target = torch.zeros((batch_size, seq_len, 8), dtype=torch.long)
    
    for i, vocab_size in enumerate(n_token):
        dummy_input[:, :, i] = torch.randint(0, vocab_size, (batch_size, seq_len))
        dummy_target[:, :, i] = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    dummy_mask = torch.ones(batch_size, seq_len)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Target shape: {dummy_target.shape}")
    print(f"  Mask shape: {dummy_mask.shape}")
    
    losses = model.train_step(dummy_input, dummy_target, dummy_mask)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Loss values:")
    loss_names = ['Family', 'Bar', 'Pitch', 'Velocity', 'Duration', 'Chord', 'Rest', 'Tempo']
    for name, loss in zip(loss_names, losses):
        print(f"    {name:12s}: {loss.item():.4f}")
    
    print(f"\n✓ Total loss: {sum(l.item() for l in losses):.4f}")