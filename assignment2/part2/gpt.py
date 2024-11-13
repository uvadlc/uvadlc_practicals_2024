"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple



class BERTGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class RMSNorm(nn.Module):
    """
    Implementation of the RMSNorm normalization layer. RMSNorm is a layer normalization
    technique that normalizes the input tensor using the root mean square (RMS) of the
    tensor values. This normalization technique is used in some transformer models as
    an alternative to standard layer normalization.
    Reference: Root Mean Square Layer Normalization (RMSNorm) https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the norm of the input tensor and divide by the norm
        # Scale the normalized tensor by the learned weight parameter
        return output

class CausalSelfAttention(nn.Module):
    """
    Implements a vanilla multi-head masked self-attention layer with a projection at the end, 
    designed for causal (unidirectional) attention models. This layer ensures that 
    during self-attention, a token does not attend to subsequent tokens, making it suitable for 
    tasks like language modeling.

    The self-attention mechanism is a key component in allowing the model to focus on different 
    parts of the input sequence when making predictions. This implementation includes a causal mask 
    to ensure the autoregressive property in models like GPT.

    Attributes:
        c_attn (nn.Linear): Linear layer for combined key, query, and value projections.
        c_proj (nn.Linear): Linear layer for output projection.
        attn_dropout (nn.Dropout): Dropout layer applied to attention weights.
        resid_dropout (nn.Dropout): Dropout layer applied to the output of the self-attention layer.
        bias (torch.Tensor): Causal mask to ensure attention is only applied to the left in the input sequence.
        n_head (int): Number of attention heads.
        n_embd (int): Dimensionality of the embeddings/hidden states.

    Parameters:
        config (object): Configuration object with attributes n_embd, n_head, attn_pdrop, resid_pdrop, 
                         and block_size. n_embd is the embedding dimension, n_head is the number of 
                         attention heads, attn_pdrop is the dropout probability for the attention, 
                         resid_pdrop is the dropout probability for the output, and block_size is the 
                         size of the causal mask.
    """

    def __init__(self, config, debug = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_flash_attn = config.use_flash_attn

        # Frequency for RoPE
        dim = config.n_embd // config.n_head
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        self.config  = config 
        self.debug = debug

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embeddings using sine and cosine functions to the query and key tensors.
        
        Args:
            xq (torch.Tensor): Query tensor of shape [batch, num_heads, seq_len, head_dim].
            xk (torch.Tensor): Key tensor of shape [batch, num_heads, seq_len, head_dim].
            pos (torch.Tensor): Sinusoidal position embeddings for RoPE of shape [1, 1, seq_len, head_dim].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the modified query and key tensors.
        """
        # Generate RoPE embeddings dynamically based on T
        seq_pos = ...  # Shape: (T)
        freqs = ...    # Shape: (T, dim // 2)
        pos_emb = ...  # Shape: (1, 1, T, dim)
        
        # Split pos into sin and cos components, repeating each to match xq and xk dimensions
        pos_sin = ...
        pos_cos = ...
        
        # Apply RoPE transformation: pair and rotate dimensions
        # Rotate query and key tensors
        xq_rot = ...
        xk_rot = ...
        raise NotImplementedError
        
        return xq_rot, xk_rot
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Split output of attention-head in query, key and value
        q, k ,v  = ...

        q = ...
        k = ...
        v = ...

        if not self.config.abs_emb:
            q, k = self.apply_rotary_emb(q, k, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Calculate attention weights using key and queries, moreover, apply dropout to the weigths
        # Mask the calculated attention weights with the mask parameter.

        if self.use_flash_attn:
            y = ...
        else:
            # Compute attention scores
            att = ... 
            # Apply causal mask
            # Apply attention to the values
            y = ... # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y if not self.debug else {"att_probs": att, "q": q, "k": k, "v": v}


class TransformerDecoderBlock(nn.Module):
    """
    Represents a single decoder layer of a Transformer model, encapsulating a layer of causal self-attention 
    followed by a feed-forward neural network (MLP). This is a fundamental component in 
    Transformer-based models, especially those used for tasks that require understanding the 
    sequential or temporal relationships in data, like language modeling.

    The decoder layer applies layer normalization before the self-attention and the MLP to stabilize 
    the learning process. The MLP itself consists of two linear transformations with a GELU 
    activation in between.

    Attributes:
        layer_norm_1 (RMSNorm): Layer normalization applied before the self-attention layer.
        self_attention (CausalSelfAttention): The causal self-attention layer.
        layer_norm_2 (RMSNorm): Layer normalization applied before the MLP.
        mlpf (nn.Sequential): A feedforward pass through the MLP with a Linear (output=4*n_embd), GELU non-linearity(use the BERTGELU), Linear (output=n_embd), and residual Dropout.

    Parameters:
        config (object): Configuration object with attributes n_embd and resid_pdrop. n_embd is the 
                         embedding dimension, and resid_pdrop is the dropout probability for the 
                         output of the MLP.
    """
    def __init__(self, config):
        super().__init__()
        # Initialize the layers
        raise NotImplementedError
    def forward(self, x):
        # Forward pass through the Decoder Layer
        out = ...
        return out


class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = Namespace()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.use_flash_attn = False
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.block_size = config.block_size

        # Check whether either the type is given or the params are given. (With XOR)
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        print(type_given, params_given)
        assert type_given ^ params_given

        # translate from model_type to detailed configuration
        if type_given:
            config.__dict__.update(
                {
                    # names follow the huggingface naming conventions
                    # GPT-1
                    "openai-gpt": dict(
                        n_layer=12, n_head=12, n_embd=768
                    ),  # 117M params
                    # GPT-2 configs
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    "gpt2-medium": dict(
                        n_layer=24, n_head=16, n_embd=1024
                    ),  # 350M params
                    "gpt2-large": dict(
                        n_layer=36, n_head=20, n_embd=1280
                    ),  # 774M params
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # Some extra tiny models
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=384),  # 192
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                }[config.model_type]
            )

        # Creation transformer
        self.transformer = nn.ModuleDict(dict(
            w_token_emb = nn.Embedding(config.vocab_size, config.n_embd),
            w_pos_emb = nn.Embedding(config.block_size, config.n_embd), #in this assignment, you have to instead use the rotary positional embeddings, but we keep the placeholder if you want to use the pre-trained model
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([TransformerDecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "No pretrained weights available for specified model-type.. Choose between 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx: torch.Tensor):
        """ Processes a batch of word indices through the transformer model to generate logits. This function takes a batch of 
        word indices, applies word and position embeddings, and then forwards the data through the transformer's layers to 
        produce logits. It is typically used during the forward pass of a neural network in training or evaluation.

        Parameters:
            - idx (torch.Tensor): A tensor of word indices with shape (batch_size, sequence_length). The word 
                                  indices should be integers representing words in the model's vocabulary.

        Returns:
            - torch.Tensor: The logits output by the model, representing the unnormalized probabilities for each word in the 
                            vocabulary at each position in the sequence. The shape of the logits tensor is 
                            (batch_size, sequence_length, vocabulary_size).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # Forward token and position embedders
        # token embeddings of shape (b, t, n_embd)
        # apply dropout to the tokens
        tok_emb = ...

        if self.config.abs_emb:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            pos_emb = self.transformer.w_pos_emb(pos) 
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Iterate through the transformer blocks
        # Apply final layer normalization and linear layer to produce logits
        logits = ...

        return logits

    @torch.inference_mode()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int, temperature:float = 1.0, do_sample:bool = False, top_k:int = None, top_p: float = 0.6):
        """
        Generates a sequence of tokens by autoregressively predicting new tokens based on the 
        provided context (idx). The generation process can be controlled by temperature, sampling 
        strategy, and a top-k filtering of the logits.

        This method is typically used in a language model to extend a given sequence of token indices 
        with new, plausible tokens. It's important to use this method in the `eval()` mode of the model 
        to disable dropout and other training-specific behaviors for more predictable outputs.

        Parameters:
            idx (torch.LongTensor): A tensor of token indices of shape (batch size, sequence length) 
                                    used as the initial context for generation.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float, optional): A scaling factor to control the randomness of predictions by 
                                            scaling the logits before applying softmax. Higher values 
                                            increase diversity, lower values make the model more confident 
                                            in its top choices. Default is 1.0.
            do_sample (bool, optional): If True, samples from the probability distribution of the 
                                        predicted tokens, otherwise takes the most likely token. 
                                        Default is False.
            top_k (int, optional): If set, only the top-k most likely next tokens are considered for 
                                    sampling at each step. If None, all tokens are considered. 
                                    Default is None.
            top_p (float, optional): If set, only the most likely tokens whose cumulative probability 
                                    mass is less than p are considered for sampling at each step. 
                                    If None, all tokens are considered. Default is 0.6.

        Returns:
            torch.LongTensor: The tensor of token indices including the original and the newly generated 
                                tokens, with shape (batch size, sequence length + max_new_tokens).
        """
        assert not (top_k and top_p), "You can only use one of top_k or top_p sampling"
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # forward the model to get the logits for the index in the sequence
            # pluck the logits at the final step and scale by desired temperature

            if not do_sample:
                # take the most likely token
                idx_next = ...
            
            else:
                # pluck the logits at the final step and scale by desired temperature

                # apply softmax to convert logits to (normalized) probabilities

                # optionally only consider top-k logits for sampling. 
                if top_k is not None:
                    pass

                # optionally apply top-p sampling
                if top_p is not None:
                    pass
            
            # append sampled index to the running sequence and continue
            idx = ...

        return idx
