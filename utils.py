def get_params(model):
    '''
    Returns the parameters of a model as a single vector
    Args:
        model - instance of torch.nn.Module
    Returns:
        params - list of shape [n_params]
    '''
    params = []
    for param in model.parameters():
        params.extend(param.flatten().detach().tolist())
    return params