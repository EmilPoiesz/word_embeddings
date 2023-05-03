import torch
from datetime import datetime

def train(n_epochs, optimizer, model, loss_fn, train_loader, device):

    n_batch = len(train_loader)
    losses_train = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, n_epochs + 1):

        loss_train = 0.0
        for contexts, targets in train_loader:

            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            outputs = model(contexts)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()

        losses_train.append(loss_train / n_batch)

        if epoch == 1 or epoch % 5 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.5f}'.format(
                datetime.now().time(), epoch, loss_train / n_batch))
    return losses_train

def compute_cosine_sim(model, loader, device):
    """
    Computes accuracy as average cosine similarity between predicted and target embedding
    """
    model.eval()
    cosine_sim = torch.nn.CosineEmbeddingLoss(reduction='mean')
    n_batch = len(loader)

    cos_loss = 0

    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            outputs = model(contexts)
            _, predicted = torch.max(outputs, dim=1)
            cos_loss += cosine_sim(predicted, targets, target=torch.tensor(1))

    sim = cos_loss / n_batch
    return sim