
def df_to_data_loader(df, target_column='CLPVariation'):
    import torch
    import torch.utils.data as data_utils

    train, test = train_test_split(df, test_size=0.2)

    # Creating np arrays
    data_target = df[target_column].values
    data_features = df.drop(target_column, axis='columns').values

    # Passing to DataLoader
    data_tensor = data_utils.TensorDataset(torch.Tensor(data_features), torch.Tensor(data_target))
    data_loader = data_utils.DataLoader(data_tensor, batch_size=10, shuffle=True)

    return data_loader


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


def validate(model, val_loader):
    model.eval()

    for i, data in enumerate(val_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Make predictions for this batch
        outputs = model(inputs)
        print("Validation: ")


def test(model, test_loader):
    model.eval()

    for i, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Make predictions for this batch
        outputs = model(inputs)
        print(f"{outputs=}, {labels=}")
