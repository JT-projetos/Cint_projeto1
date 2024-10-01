
def create_data_loaders(df):
    import torch
    import torch.utils.data as data_utils
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.2)

    # Creating np arrays
    train_target = train['CLPVariation'].values
    train_features = train.drop('CLPVariation', axis='columns').values

    test_target = test['CLPVariation'].values
    test_features = test.drop('CLPVariation', axis='columns').values

    # Passing to DataLoader
    train_tensor = data_utils.TensorDataset(torch.Tensor(train_features), torch.Tensor(train_target))
    train_loader = data_utils.DataLoader(train_tensor, batch_size=10, shuffle=True)

    test_tensor = data_utils.TensorDataset(torch.Tensor(train_features), torch.Tensor(train_target))
    test_loader = data_utils.DataLoader(test_tensor, batch_size=10, shuffle=True)

    return train_loader, test_loader


def train_one_epoch(model, optimizer, loss_fn, train_loader):
    for i, data in enumerate(train_loader, 0):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        #print(f"{inputs.shape=}, {labels.shape=}, {outputs.shape=}")
        # Compute the loss and its gradients
        labels = labels.unsqueeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()


def test(model, test_loader):
    model.eval()

    for i, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Make predictions for this batch
        outputs = model(inputs)
        print(f"{outputs=}, {labels=}")
