#%%
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx

class LinearRNNUnit(eqx.Module):
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
    Win: eqx.nn.Embedding
    Wrec: jax.Array
    in_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, *, init_scale=0.1, key):
        in_key, rec_key = jr.split(key, num=2)
        self.in_size = in_size
        self.hidden_size = hidden_size

        # Create embedding module
        self.Win = eqx.nn.Embedding(
            num_embeddings = self.in_size, 
            embedding_size = self.hidden_size,
            key = in_key)

        # Create recurrent connectivity matrix as a scaled orthogonal matrix
        self.Wrec = init_scale * jr.orthogonal(rec_key, hidden_size)

    def __call__(self, inputs):
        """
        Run the linear RNN over the time dimension of inputs.

        inputs: array of shape [time_steps, in_size]
        Returns: array of hidden states shape [time_steps, hidden_size]
        """
        
        inputs_embedded = jax.vmap(self.Win)(inputs)

        def f(carry, inp):
            # Compute next hidden state
            h = self.Wrec @ carry + inp
            return h, h

        # Initialize carry to zeros with the correct hidden_size
        init_carry = jnp.zeros((self.hidden_size,))

        # Scan over time dimension
        _, hiddens = jax.lax.scan(f, init_carry, inputs_embedded)
        return hiddens

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
                 *, key):

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

class LinearRNN(eqx.Module):
    """
    Full Linear RNN with readout layer

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
    rnn: LinearRNNUnit
    readout: Readout

    def __init__(self, 
                 in_size, 
                 hidden_size, 
                 out_size, 
                 readout_depth, 
                 *, 
                 rnn_scale = 0.1,
                 key):
        
        rnn_key, readout_key = jr.split(key, 2)
        self.rnn = LinearRNNUnit(in_size, 
                                 hidden_size,
                                 init_scale=rnn_scale, 
                                 key = rnn_key)
        
        self.readout = Readout(
            depth = readout_depth, 
            in_size = hidden_size, 
            hidden_size = hidden_size, 
            out_size = out_size, 
            key = readout_key)
        
    def __call__(self, x):
        x = self.rnn(x)
        return jax.vmap(self.readout)(x)