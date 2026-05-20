#%%
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.experimental.sparse as sparse
from jax.experimental.sparse import BCOO

class dyck_hmm():
    def __init__(self, k, m, build_full_model=False):
        self.k = k
        self.m = m

        # Total number of tree states, plus one absorbing terminal state.
        self.num_states = (k**(m + 1) - 1) // (k - 1)
        self.terminal_state = self.num_states
        self.internal_cutoff = self.num_states // self.k

        # Only build this if you actually need filtering.
        self.K_dyck = None
        self.E_dyck = None
        self.T_dyck = None
        if build_full_model:
            self.K_dyck, self.E_dyck, self.T_dyck = self.get_full_model()

    def _parent_state_and_obs(self, state):
        terminal = self.terminal_state

        parent_state = jnp.where(
            state == terminal,
            terminal,
            jnp.where(state == 0, terminal, (state - 1) // self.k),
        )
        parent_obs = jnp.where(
            state == terminal,
            2 * self.k + 1,
            jnp.where(state == 0, 2 * self.k, ((state - 1) % self.k) + self.k),
        )
        return parent_state, parent_obs

    def _dyck_step(self, old_state, key, min_length, idx):
        terminal = self.terminal_state
        parent_state, parent_obs = self._parent_state_and_obs(old_state)

        has_children = old_state < self.internal_cutoff
        is_terminal = old_state == terminal

        child_probs = jnp.where(
            has_children,
            jnp.full((self.k,), 0.5 / self.k, dtype=jnp.float32),
            jnp.zeros((self.k,), dtype=jnp.float32),
        )
        parent_prob = jnp.where(has_children, 0.5, 1.0)
        parent_prob = jnp.where(
            jnp.logical_and(old_state == 0, idx < min_length), 
            0.0, 
            parent_prob)

        probs = jnp.concatenate([child_probs, 
                                 jnp.array([parent_prob], dtype=jnp.float32)])
        probs = jnp.where(
            is_terminal,
            jnp.concatenate([jnp.zeros((self.k,), dtype=jnp.float32), 
                             jnp.array([1.0], dtype=jnp.float32)]),
            probs,
        )
        probs = probs / probs.sum()

        choice = jr.choice(key, self.k + 1, p=probs)

        next_state = jnp.where(choice < self.k, 
                               old_state * self.k + choice + 1, 
                               parent_state)
        observation = jnp.where(choice < self.k, choice, parent_obs)

        next_state = jnp.where(is_terminal, terminal, next_state)
        observation = jnp.where(is_terminal, 2 * self.k + 1, observation)

        return next_state, observation

    def sample_sequence(self, num_timesteps, min_length, key):
        def f(carry, key):
            (old_state, idx), _ = carry
            new_state, observation = self._dyck_step(
                old_state, key, min_length, idx)
            new_carry = (new_state, idx+1)

            return (new_carry, observation), (new_carry, observation)

        keys = jr.split(key, num_timesteps)
        _, ((states, _), obs) = jax.lax.scan(f, ((0, 0), 0), keys)
        return states, obs

    def batch_sample_sequence(self, batch_size, num_timesteps, min_length, 
                              *, key):
        if not isinstance(min_length, jax.Array):
            min_length = jnp.full(batch_size, min_length)

        states, sequences = jax.vmap(
            self.sample_sequence, in_axes=[None, 0, 0]
        )(num_timesteps, min_length, jr.split(key, batch_size))

        return states, sequences

    # ==== Filtering code ====
    def get_full_model(self):
        """
        Build sparse transition/emission tensors directly from the sampling
        rules used in _parent_state_and_obs and _dyck_step.
        """
        num_states = self.num_states + 1  # include terminal state

        transition_rows = []
        transition_cols = []
        transition_probs = []
        emission_obs = []

        for state in range(num_states):
            state = int(state)

            # Terminal state is absorbing.
            if state == self.terminal_state:
                transition_rows.append(state)
                transition_cols.append(state)
                transition_probs.append(1.0)
                emission_obs.append(2 * self.k + 1)
                continue

            parent_state, parent_obs = self._parent_state_and_obs(jnp.asarray(state))
            parent_state = int(parent_state)
            parent_obs = int(parent_obs)

            # Internal nodes: k children with prob 0.5/k, parent with prob 0.5.
            if state < self.internal_cutoff:
                for child_idx in range(self.k):
                    child_state = state * self.k + child_idx + 1
                    transition_rows.append(state)
                    transition_cols.append(child_state)
                    transition_probs.append(0.5 / self.k)
                    emission_obs.append(child_idx)

                transition_rows.append(state)
                transition_cols.append(parent_state)
                transition_probs.append(0.5)
                emission_obs.append(parent_obs)

            # Leaves: only parent transition with prob 1.
            else:
                transition_rows.append(state)
                transition_cols.append(parent_state)
                transition_probs.append(1.0)
                emission_obs.append(parent_obs)

        transition_rows = jnp.asarray(transition_rows, dtype=jnp.int32)
        transition_cols = jnp.asarray(transition_cols, dtype=jnp.int32)
        transition_probs = jnp.asarray(transition_probs, dtype=jnp.float32)
        emission_obs = jnp.asarray(emission_obs, dtype=jnp.int32)

        T_indices = jnp.stack([transition_rows, transition_cols], axis=1)
        E_indices = jnp.stack([emission_obs, transition_rows, transition_cols], axis=1)

        T_dyck = BCOO((transition_probs, T_indices),
                      shape=(num_states, num_states))

        E_dyck = BCOO((jnp.ones_like(transition_probs), E_indices),
                      shape=(2 * self.k + 2, num_states, num_states))

        K_dyck = BCOO((transition_probs, E_indices),
                      shape=(2 * self.k + 2, num_states, num_states))

        return K_dyck, E_dyck, T_dyck
    
    def filter(self, sequence):
        num_states = self.K_dyck.shape[-1]

        # ======== Define filter update equations ========
        def _dyck_update(a, y_t, K):
            return a @ K[y_t]
        dyck_update = lambda a, y: _dyck_update(a, y, self.K_dyck)

        def f(alpha, input):
            next_alpha = dyck_update(alpha, input)
            return next_alpha / next_alpha.sum(), next_alpha / next_alpha.sum()

        alpha0 = jnp.zeros(num_states).at[0].set(1).astype(jnp.float32)
        _, filtered_alphas = jax.lax.scan(f, init=alpha0, xs=sequence)

        return filtered_alphas

    def batch_filter(self, sequences):
        return jax.vmap(self.filter)(sequences)
    
    # ==== One Step Prediction ====
    def one_step_prediction(self, sequence):
        alpha = self.filter(sequence)
        return (alpha @ self.K_dyck).sum(2)

    def batch_one_step_prediction(self, sequences):
        return jax.vmap(self.one_step_prediction)(sequences)
    
class eps_softened_dyck_hmm():
    def __init__(self, k, m):
        self.k = k
        self.m = m

        # Compressed model for Dyck languages 
        dyck_neighbors, dyck_transition_probs, dyck_observations = \
            self.get_compressed_dyck_model()
        self.compressed_dyck_model = {
            'neighbors': dyck_neighbors,
            'transition_probs': dyck_transition_probs,
            'observations': dyck_observations}

        # Compressed model for Null languages 
        null_neighbors, null_transition_probs = \
            self.get_null_model()
        self.compressed_null_model = {
            'neighbors': null_neighbors,
            'transition_probs': null_transition_probs,
            'observations': dyck_observations,
            'observation_probs': dyck_transition_probs.at[0,-1].set(0)}
        
        self.K_dyck, self.E_dyck, self.T_dyck = self.get_full_model()
        
    # ==== Sampling code ====
    # Generate relationships for models
    def get_state_relationships(self):
        num_states = int((self.k**(self.m+1) - 1) / (self.k-1))
        num_classes = 2*self.k

        # children_vec[i] = children of node i
        def get_child_vec(node):
            return self.k*node + jnp.arange(self.k) + 1

        idx = num_states // self.k

        children_vec = jnp.full((num_states + 1, self.k), -1, dtype=jnp.int32)
        children_vec = children_vec.at[:idx].set(
            jax.vmap(get_child_vec)(jnp.arange(idx))
        )

        # parent_vec[i,:] = parent of node i, corresponding close index
        parents_vec = jnp.zeros((num_states + 1, 2), dtype=jnp.int32)
        parents_vec = parents_vec.at[1:num_states, 0].set(
            (jnp.arange(1, num_states) - 1) // self.k)
        parents_vec = parents_vec.at[1:num_states, 1].set(
            ((jnp.arange(1, num_states) - 1) % self.k) + self.k)
        parents_vec = parents_vec.at[0,:].set([num_states, num_classes])
        parents_vec = parents_vec.at[-1,:].set([num_states, num_classes+1])

        return children_vec, parents_vec
    
    def get_compressed_dyck_model(self):
        '''
        Generates compressed relational structure for standard Dyck-(k, m) 
        langauges and null model (equal probability to transition to non-terminal 
        states). Does not explicitly generate full transition and emission matrices
        as these are too cumbersome to use for sampling sequences for large models
        e.g., k >= 8.

        INPUTS:
        children_vec, parents_vec - Outputs from get_state_relationships(k, m)

        RETURNS:
        dyck_neighbors   - Array where dyck_neighbors[i] = array of neighbors 
                           of state i. Padded with -1 (Num_states x k+1)
        dyck_trans_probs - Array where dyck_neighbors_probs[i,j] = probability
                             of transitioning from state i to the j-th neighbor
                             from dyck_neighbors (Num_states x k+1)
        dyck_obs_probs   - Array where dyck_neighbors_probs[i,j] = observation
                             associated with the transition from state i to the 
                             j-th neighbor from dyck_neighbors (Num_states x k+1)
        '''

        children_vec, parents_vec = self.get_state_relationships()
        num_states = len(children_vec)

        # Dyck Languages neighbors
        def get_dyck_neighbors(state):
            children, children_idx = children_vec[state], jnp.arange(self.k)
            parent, parent_idx = parents_vec[state]

            # Neighboring states
            neighbors = jnp.append(children, parent)

            # Probability of transitioning to that state
            children_probs = jnp.maximum(
                jnp.where(children < 0, children, 0.5/self.k), 0)
            parent_probs = jnp.ones(1) - children_probs.sum()
            next_state_probs = jnp.append(children_probs, parent_probs)
            
            # Observation when transtioning between states
            observations = jnp.append(children_idx, parent_idx)
            return neighbors, next_state_probs, observations

        neighbors, transition_probs, observation_probs = \
            jax.vmap(get_dyck_neighbors)(jnp.arange(num_states))

        return neighbors, transition_probs, observation_probs, \
            
    def get_null_model(self):
        # TODO: this needs to be done better. Both Neighbors and 
        #       transition_probsis very redundant and

        num_states = int((self.k**(self.m+1) - 1) / (self.k-1)) + 1

        # === Null Model ===
        # dyck_observation_probs is the same as dyck_neighbors_obs
        neighbors = jnp.vstack(
            (jnp.tile(jnp.arange(num_states), (num_states-1,1)),
            jnp.full(num_states, -1).at[-1].set(num_states-1))
        )

        transition_probs = jnp.vstack((
            jnp.full(num_states, 1/(num_states)),
            jnp.full((num_states-2, num_states), 
                    1/(num_states-1)).at[:,-1].set(0),
            jnp.full(num_states, 0).at[-1].set(1))
        )

        return neighbors, transition_probs
    
    # Sample functions
    def _dyck_step(self, old_state, key, 
                neighbors, next_state_probs, obs_vec,
                min_length, idx):
        
        p = jax.lax.cond(idx < min_length,
                    lambda x: next_state_probs.at[0, -1].set(0)[old_state],
                    lambda x: next_state_probs[old_state],
                    idx)
        
        i = jr.choice(key, obs_vec.shape[1], p = p)
        next_state = neighbors[old_state, i]
        observation = obs_vec[old_state, i]

        return next_state, observation

    def _null_step(self, old_state, key, neighbors, next_state_probs, 
                obs_vec, obs_probs,
                min_length, idx):
        terminal_state = neighbors[0,-1]
        
        # Do not allow transitions into the absorbing state until min_length
        # is reached
        transition_probs = jax.lax.cond(
            idx < min_length,
            lambda x: next_state_probs.at[0, -1].set(0)[old_state],
            lambda x: next_state_probs[old_state],
            idx)
        
        t_key, e_key = jr.split(key, 2)
        next_state = jr.choice(t_key, 
                        neighbors[old_state], 
                        p=transition_probs)

        emission_probs = jax.lax.cond(
            jnp.logical_and(old_state == 0, next_state == terminal_state),
            lambda x: jnp.zeros(self.k + 1).at[-1].set(1),
            lambda x: obs_probs[old_state],
            idx)
        observation = jr.choice(e_key, 
                                obs_vec[old_state], 
                                p = emission_probs)
        
        return next_state, observation
        
    def sample_sequence(self, epsilon, num_timesteps, min_length, key):
        # Define sampling steps for true and null steps of specific models
        dyck_step = lambda old_state, key, idx: \
            self._dyck_step(old_state, key, 
                    self.compressed_dyck_model["neighbors"], 
                    self.compressed_dyck_model["transition_probs"], 
                    self.compressed_dyck_model["observations"],
                    min_length, idx)

        null_step = lambda old_state, key, idx: \
            self._null_step(old_state, key, 
                        self.compressed_null_model["neighbors"], 
                        self.compressed_null_model["transition_probs"], 
                        self.compressed_null_model["observations"],
                        self.compressed_null_model["observation_probs"],
                        min_length, idx)
        
        # === Scan function for sampling ===
        def f(carry, key):
            (old_state, idx), _ = carry
            m = jr.uniform(key)

            new_state, observation = jax.lax.cond(
                m < epsilon, 
                null_step, dyck_step, 
                old_state, key, idx)

            new_carry = (new_state, idx+1)
            return (new_carry, observation), (new_carry, observation)

        keys = jr.split(key, num_timesteps)
        _, ((states, _), obs) = jax.lax.scan(f, ((0, 0), 0), keys)
        return states, obs
    
    def batch_sample_sequence(self, batch_size, epsilon, num_timesteps, 
                               min_length, *, key):
        
        if not isinstance(min_length, jax.Array):
            min_length = jnp.full(batch_size, min_length)
            
        states, sequences = jax.vmap(
            self.sample_sequence, in_axes=[None, None, 0, 0])(
                epsilon, num_timesteps, min_length, jr.split(key, batch_size))
        
        return states, sequences

    # ==== Filtering code ====
    def get_full_model(self):
        '''
        Generates full relational structure for standard Dyck-(k, m) langauges 
        by explicitly generate sparse transition and emission matrices. 
        Necessary for filtering. 
        
        Note: In practice, all you need for filtering is K_dyck, but we return 
        E_dyck and T_dyck as well for completeness sake.

        INPUTS:
        children_vec, parents_vec - Outputs from get_state_relationships(k, m)

        RETURNS:
        K_dyck - Tensor combining transition and emission probabilities for edge-
                emitting HMM. K_dyck[i, j, k] = probability of transitioning from
                state j to k and observing observation i
                (num_classes x num_states x num_states)
        E_dyck - Tensor defining the emission probabilities for state transitions.
                E_dyck[i,j,k] = probability of observing observation i given the 
                transition from state j to k.
                (num_classes x num_states x num_states)
        T_dyck - Matrix defining the transition probabilities for each state.
                T[i,j] = probability of transitioning from state i to state j
                (num_states x num_states)
        '''
        
        children_vec,parents_vec = self.get_state_relationships()
        num_states = len(children_vec)

        # ======== Use Tree to Generate Sparse Transition Matrix ========
        def get_transitions_with_children(node):
            down_rows = jnp.full((self.k,1), node)
            down_cols = children_vec[node]
            down_probs = jnp.full((self.k,1), 0.5 / self.k)

            up_rows = node
            up_cols = parents_vec[node][0]
            up_probs = 0.5
            return down_rows, down_cols, down_probs, up_rows, up_cols, up_probs

        def get_transitions_without_children(node):
            up_rows = node
            up_cols = parents_vec[node][0]
            up_probs = 1
            return up_rows, up_cols, up_probs

        down_rows, down_cols, down_probs, \
            up_rows1, up_cols1, up_probs1 = \
            jax.vmap(get_transitions_with_children)(
                jnp.arange((num_states - 1) // self.k))

        up_rows2, up_cols2, up_probs2 = \
            jax.vmap(get_transitions_without_children)(
                jnp.arange((num_states - 1) // self.k, num_states))

        transition_rows = jnp.concatenate(
            [down_rows.flatten(), up_rows1, up_rows2])
        transition_cols = jnp.concatenate(
            [down_cols.flatten(), up_cols1, up_cols2])
        transition_probs = jnp.concatenate([
            down_probs.flatten(), up_probs1, up_probs2])

        transitions_indices = jnp.stack([transition_rows, 
                                         transition_cols], axis=1)
        T_dyck = BCOO((transition_probs, transitions_indices), 
                    shape=(num_states, num_states))

        # ======== Use Tree to Generate Sparse Emission Matrix ========
        def get_emissions_with_children(node):
            children, children_idx = children_vec[node], jnp.arange(self.k)

            down_rows = jnp.full((self.k, 1), node)
            down_cols = children
            down_dpth = children_idx

            parent, parent_idx = parents_vec[node]
            up_rows = node
            up_cols = parent
            up_dpth = parent_idx

            return down_rows, down_cols, down_dpth, up_rows, up_cols, up_dpth

        def get_emissions_without_children(node):
            parent, parent_idx = parents_vec[node]
            up_rows = node
            up_cols = parent
            up_dpth = parent_idx

            return up_rows, up_cols, up_dpth

        down_rows, down_cols, down_dpth, \
            up_rows1, up_cols1, up_dpth1 = \
            jax.vmap(get_emissions_with_children)(
                jnp.arange((num_states - 1) // self.k))

        up_rows2, up_cols2, up_dpth2 = \
            jax.vmap(get_emissions_without_children)(
                jnp.arange((num_states - 1) // self.k, num_states))

        emission_rows = jnp.concatenate([down_rows.flatten(), 
                                         up_rows1, up_rows2])
        emission_cols = jnp.concatenate([down_cols.flatten(), 
                                         up_cols1, up_cols2])
        emission_dpth = jnp.concatenate([down_dpth.flatten(), 
                                         up_dpth1, up_dpth2])

        emission_probs = jnp.ones_like(emission_dpth)
        emission_indices = jnp.stack([emission_dpth, 
                                      emission_rows, 
                                      emission_cols], axis=1)
        E_dyck = BCOO((emission_probs, emission_indices), 
                    shape=(2*self.k + 2, num_states, num_states))

        # ======== Use T and E to Generate Sparse Edge Kernel Matrix ========
        K_dyck = BCOO((emission_probs * transition_probs, emission_indices),
                    shape=(2*self.k + 2, num_states, num_states))
        
        return K_dyck, E_dyck, T_dyck

    def filter(self, sequence, epsilon):
        num_states = self.K_dyck.shape[-1]

        # Shortcut 
        p_null = sparse.bcoo_reduce_sum(self.K_dyck, axes=[2])

        # ======== Define filter update equations ========
        def _dyck_update(a, y_t, K):
            return a @ K[y_t]
        dyck_update = lambda a, y: _dyck_update(a, y, self.K_dyck)

        def _null_update(alpha, y_t, p_null, num_states):
            t0 = 1 / num_states
            t_rest = 1 / (num_states-1)

            alpha2 = alpha * p_null[y_t].todense()
            sum_all   = alpha2.sum()
            sum_inner = alpha2[:-1].sum()

            weights = jnp.array([
                t0 * sum_all,
                t_rest * sum_inner,
                alpha2[-1]
            ])

            inner = jnp.full(num_states - 2, weights[1])
            return jnp.concatenate([weights[:1], inner, weights[2:]])
        null_update = lambda a, y: _null_update(a, y, p_null, num_states)

        def f(alpha, input):
            dyck_alpha = dyck_update(alpha, input)
            null_alpha = null_update(alpha, input)

            next_alpha = (1-epsilon) * dyck_alpha + epsilon * null_alpha
            return next_alpha / next_alpha.sum(), next_alpha / next_alpha.sum()

        alpha0 = jnp.zeros(num_states).at[0].set(1)
        _, filtered_alphas = jax.lax.scan(f, init=alpha0, xs=sequence)

        return filtered_alphas

    def batch_filter(self, sequences, epsilon):
        return jax.vmap(HMM.filter, in_axes=[0, None])(
            sequences, epsilon)
    
    # ==== One Step Prediction ====
    def one_step_prediction(self, sequence, epsilon):
        alpha = self.filter(sequence, epsilon)
        p_null = sparse.bcoo_reduce_sum(self.K_dyck, axes=[2])

        dyck_pred = (alpha @ self.K_dyck).sum(2)
        null_pred = p_null @ alpha.T

        return (1-epsilon) * dyck_pred + epsilon * null_pred

    def batch_one_step_prediction(self, sequences, epsilon):
        return jax.vmap(self.one_step_prediction, in_axes=[0, None])(
            sequences, epsilon)

if __name__ == "__main__":
    HMM = dyck_hmm(32, 5)

    
# %%

