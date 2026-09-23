import torch


def ema_update(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)
        for eb, pb in zip(ema.buffers(), model.buffers()):
            eb.copy_(pb)
