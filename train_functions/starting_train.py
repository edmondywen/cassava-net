import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
import tensorboard as tf 
import datetime 

#have a parameter for "how often to log data to TensorBoard"
def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders - creates a "list of batches" (sort of)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )


    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    # Init totals/correct for evaluating train accuracy 
    step = 0
    correct = 0
    total = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # Backpropagation and gradient descent
            input_data, labels = batch
            predictions = model.forward(input_data)
            loss = loss_fn(predictions, labels)
            predictions = predictions.argmax(axis = 1) 
#argmax gives the max value in an axis. softmax has "soft" values (ie a range of probabilities). not 100% sure
            total += len(labels) #have to add entire batch size
            correct += (predictions == labels).sum().item()
            accuracy = correct/total #double check if this is correct 

            loss.backward()
            optimizer.step()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar("train_loss", loss, global_step = step)
                writer.add_scalar("train_accuracy", accuracy, global_step = step)
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                accuracy, loss = evaluate(val_loader, model, loss_fn)
                writer.add_scalar("validation_loss", loss, global_step = step)
                writer.add_scalar("validation_accuracy", accuracy, global_step = step)
                torch.save(model.state_dict(), './model.pt')
            
            optimizer.zero_grad()
            step += 1

        print("Epoch " , epoch, " Loss ", loss)


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    correct = 0
    total = 0
    model.eval() #put network in eval mode
    for i, data in enumerate(val_loader): 
        input_data, labels = data
        predictions = model.forward(input_data)
        loss = loss_fn(predictions, labels)
        predictions = predictions.argmax(axis = 1) #trying to get 32,1: apply to axis 1 which is rows 
        
        #print(predictions.shape)
        total += len(labels)
        correct += (predictions == labels).sum().item() #boolean check. if true add one false = 0. python magic go brr
        break
    
    accuracy = correct/total
    return accuracy, loss 
