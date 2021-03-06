import argparse
import os
import time


import constants
from datasets.CNNDataset import CNNDataset
from networks.ConvNetwork import ConvNetwork
from train_functions.starting_train import starting_train


SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size}

    # Create path for training summaries
    label = f"cassava__{int(time.time())}"
    summary_path = f"{SUMMARIES_PATH}/{label}"
    os.makedirs(summary_path, exist_ok=True)
    model_path = './'
   

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    train_dataset = CNNDataset()
    val_dataset = CNNDataset(isTrain = False)
    model = ConvNetwork(3, 5) #change second arg, output_dim
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
        model_path=model_path
    )


def parse_arguments():
    parser = argparse.ArgumentParser() #this is a built in python class
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument(
        "--n_eval", type=int, default=constants.N_EVAL,
    )
    parser.add_argument("--learning_rate", type=float, default=constants.BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
