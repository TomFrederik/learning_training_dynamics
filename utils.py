

# helper functions for faster training
# from https://arxiv.org/pdf/2009.09457.pdf

def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def make_norm(state):
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size] 
        return max(rms_norm(y), rms_norm(adj_y))
    return norm

