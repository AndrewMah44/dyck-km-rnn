import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

def causal_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

class AttentionBlock(eqx.Module):
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(self, input_dim, hidden_dim, num_heads, p_dropout=0.1, 
                 *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.layernorm1 = eqx.filter_vmap(eqx.nn.LayerNorm(input_dim))
        self.layernorm2 = eqx.filter_vmap(eqx.nn.LayerNorm(input_dim))

        self.attention = eqx.nn.MultiheadAttention(
            num_heads = num_heads,
            query_size = input_dim,
            key = key1
        )

        self.linear1 = eqx.filter_vmap(
            eqx.nn.Linear(input_dim, hidden_dim, key=key2))
        self.linear2 = eqx.filter_vmap(
            eqx.nn.Linear(hidden_dim, input_dim, key=key3))

        self.dropout1 = eqx.nn.Dropout(p_dropout)
        self.dropout2 = eqx.nn.Dropout(p_dropout)

    def __call__(self, x, key, inference=False):
        key1, key2, key3 = jr.split(key, 3)
        mask = causal_mask(x.shape[0])

        # Attention
        input_x = self.layernorm1(x)
        attn = self.attention(input_x, input_x, input_x, 
                              mask=mask, key=key1, inference=inference)
        x = x + self.dropout1(attn, key=key2, inference=inference)

        # Feedforward
        h = self.layernorm2(x)
        x_ff = self.linear1(h)
        x_ff = jax.nn.gelu(x_ff)
        x_ff = self.linear2(x_ff)
        x_out = x + self.dropout2(x_ff, key=key3, inference=inference)

        return x_out
    
class TinyTransformer(eqx.Module):
    attention_blocks: list[AttentionBlock]
    token_embedding: eqx.nn.Embedding
    position_embedding: jax.Array
    final_norm: eqx.nn.LayerNorm
    lm_head: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    num_blocks: int = eqx.field(static=True)

    def __init__(self,
                 vocab_size, 
                 num_blocks, 
                 num_heads,
                 embedding_dim, 
                 hidden_dim,
                 max_length,
                 p_dropout=0.1,
                 *, key):
        self.num_blocks = num_blocks

        key1, key2, *key3, key4 = jr.split(key, num_blocks+3)

        self.token_embedding = eqx.nn.Embedding(
            vocab_size, embedding_dim, key=key1)
        self.position_embedding = 0.02 * jr.normal(
            key2, (max_length, embedding_dim))

        self.attention_blocks = [
            AttentionBlock(embedding_dim, 
                            hidden_dim,
                            num_heads, 
                            p_dropout,
                            key = key3[_])
            for _ in range(num_blocks)
        ]
        
        self.dropout = eqx.nn.Dropout(p_dropout)

        self.final_norm = eqx.nn.LayerNorm(embedding_dim)
        self.lm_head = eqx.nn.Linear(embedding_dim, vocab_size, key=key4)

    def forward(self, x, key, inference=False):
        dropout_key, *attention_keys = jr.split(key, self.num_blocks+1)

        h = jax.vmap(self.token_embedding)(x)
        h = h + self.position_embedding[:x.shape[0]]
        h = self.dropout(h, key=dropout_key, inference=inference)

        for attn_block, attn_key in zip(self.attention_blocks, 
                                        attention_keys):
            h = attn_block(h, attn_key, inference)

        h = jax.vmap(self.final_norm)(h)
        logits = jax.vmap(self.lm_head)(h)

        return logits
    
    def __call__(self, x_batch, key, inference=False):
        batch_size = x_batch.shape[0]

        return jax.vmap(self.forward, in_axes=[0, 0, None])(
            x_batch, jr.split(key, batch_size), inference)

# %%

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d_model = 16
    num_heads = 5
    d_ff = 64
    vocab_size = 5 
    num_seqs = 100
    seq_len = 50

    f = TinyTransformer(vocab_size = 5, 
                        num_blocks = 8, 
                        num_heads = 4,
                        embedding_dim = 18, 
                        hidden_dim = 16,
                        max_length = seq_len,
                        key = jr.PRNGKey(1))


    seq = jr.randint(minval=0, maxval=5, 
                     shape=(num_seqs, seq_len), key=jr.PRNGKey(1))
    
    f(seq, jr.PRNGKey(1)).shape
    

# %%
