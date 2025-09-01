import torch, os
from sklearn.metrics import f1_score

def save_ckpt(model, path): os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(model.state_dict(), path)
def action_accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()
def person_f1(logit, y):
    pred = (torch.sigmoid(logit) > 0.5).long().cpu().numpy()
    return f1_score(y.cpu().numpy(), pred)
