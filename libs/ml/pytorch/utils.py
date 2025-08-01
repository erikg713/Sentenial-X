import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
