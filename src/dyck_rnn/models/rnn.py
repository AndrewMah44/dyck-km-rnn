import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

# ==== Readouts ====
class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm

    def __init__(self, in_size, hidden_size, out_size, *, key):
        k1, k2 = jr.split(key, 2)

        self.linear1 = eqx.nn.Linear(in_size, 
                                     hidden_size, 
                                     key=k1)
        self.linear2 = eqx.nn.Linear(hidden_size,
                                     out_size,
                                     key=k2)
        self.layer_norm = eqx.nn.LayerNorm(shape=(out_size,))

    def __call__(self, x, *, key=None):
        x = self.linear1(x)
        x = jax.nn.softplus(x)
        x = self.linear2(x)
        x = self.layer_norm(x)

        return x
    
class Readout(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(self, 
                 depth, 
                 in_size, 
                 hidden_size, 
                 out_size, 
                 *, 
                 key):

        linear_key, *mlp_keys = jr.split(key, depth+1)
        mlp_layers = [
            MLP(in_size, hidden_size, hidden_size, key=key)
            for key in mlp_keys
        ]

        linear_layer = [eqx.nn.Linear(hidden_size, 
                                      out_size, 
                                      key = linear_key)]
        self.layers = eqx.nn.Sequential(
            mlp_layers + linear_layer
        )

    def __call__(self, x):
        return self.layers(x)

# ==== Linear RNNs ====
class LinearRecurrentCell(eqx.Module):
    """
    Linear RNN Unit: h_{t+1} = Wrec @ h_t + Win @ u_t

    Parameters:
        in_size: number of tokens types
        hidden_size: dimension of hidden activation
        init_scale: scale of initial RNN weights
        key: key for reproducibility

    Inputs:
        X: token sequence [num_timesteps,]
    
    Returns:
        h: hidden unit activaitons [num_timesteps,hidden_size]
    """
    W_u: eqx.nn.Embedding
    W_rec: jax.Array

    def __init__(self, in_size, hidden_size, *, init_scale=0.1, key):
        in_key, rec_key = jr.split(key, num=2)

        self.W_u = init_scale * jr.orthogonal(in_key, hidden_size)
        self.W_rec = jax.random.normal(
            rec_key, 
            (hidden_size, hidden_size)) * (1/jnp.sqrt(hidden_size))

    def __call__(self, inp, hidden):
        """
        Single step update of a linear RNN unit.

        inp:   input on trial t              [scalar]
        hidden: previous RNN unit activation [1, hidden_size]

        Returns: updated RNN unit activation [1, hidden_size]
        """
        
        return self.W_rec @ hidden + self.W_u @ inp

class RNN(eqx.Module):
    """
    RNN

    Parameters:
        in_size: number of token types
        hidden_size: dimension of hidden activation
        out_size: dimensionality of output (typically same as in_size)
        readout_depth: number of MLP layer 
            readout_depth = 0 means purely linear readout
        rnn_scale: scale of initial RNN weights
        key: key for reproducibility

    Inputs:
        X: token sequence [num_timesteps,]
    
    Returns:
        y: model outputs fed [num_timesteps,out_size]
    """
    Win: eqx.nn.Embedding
    rnn: eqx.Module
    hidden_size: int = eqx.field(static=True)
    cell_type: str = eqx.field(static=True)

    def __init__(self,
                 cell_type, 
                 vocab_size, 
                 hidden_size, 
                 *, 
                 rnn_scale = 0.1,
                 key):
        self.cell_type = cell_type.lower()        
        in_key, rnn_key = jr.split(key, 2)

        self.hidden_size = hidden_size

        self.Win = eqx.nn.Embedding(
            num_embeddings = vocab_size,
            embedding_size = hidden_size,
            key = in_key
        )

        if self.cell_type == 'linear':
            self.rnn = LinearRecurrentCell(
                hidden_size, 
                hidden_size,
                init_scale=rnn_scale, 
                key = rnn_key)
            
        elif self.cell_type == 'gru':
            self.rnn = eqx.nn.GRUCell(
                hidden_size,
                hidden_size,
                key = rnn_key)
            
        elif self.cell_type == 'lstm':
            self.rnn = eqx.nn.LSTMCell(
                hidden_size,
                hidden_size,
                key = rnn_key)
        else:
            raise ValueError(f"{cell_type} not recognized.")


    def __call__(self, inputs):
        inputs_embedded = jax.vmap(self.Win)(inputs)

        if self.cell_type == 'lstm':
            def f(carry, inp):
                carry = self.rnn(inp, carry)
                h, _ = carry

                return carry, h

            init_carry = (jnp.zeros(self.hidden_size),
                          jnp.zeros(self.hidden_size))

        else:
            def f(carry, inp):
                # Compute next hidden state
                h = self.rnn(inp, carry)

                return h, h
            
            init_carry = jnp.zeros((self.hidden_size,))

        # Scan over time dimension
        _, hiddens = jax.lax.scan(f, init_carry, inputs_embedded)
        
        return hiddens

class RecurrentSequenceModel(eqx.Module):
    rnn: RNN
    readout: Readout

    def __init__(
            self,
            cell_type, 
            vocab_size, 
            hidden_size, 
            out_size,
            readout_depth,
            *, 
            rnn_scale = 0.1,
            key):
        rnn_key, readout_key = jr.split(key, 2)

        self.rnn = RNN(
            cell_type = cell_type, 
            vocab_size = vocab_size, 
            hidden_size = hidden_size, 
            rnn_scale = rnn_scale,
            key = rnn_key)
        
        self.readout = Readout(
            depth = readout_depth, 
            in_size = hidden_size, 
            hidden_size = hidden_size, 
            out_size = out_size, 
            key = readout_key)
        
    # include key and inference inputs to make compatible with transformers...
    def __call__(self, inputs, key=None, inference=None):
        h = self.rnn(inputs)

        return jax.vmap(self.readout)(h)
