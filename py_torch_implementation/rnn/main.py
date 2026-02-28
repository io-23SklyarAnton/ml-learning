import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from py_torch_implementation.rnn import utils
from py_torch_implementation.rnn.config import Config
from py_torch_implementation.rnn.prepare_data import prepare_data
from py_torch_implementation.rnn.lstm_model import Model as LstmModel


class Trainer:
    def __init__(
            self,
            *,
            model: Module,
            dataloaders: Dict[str, DataLoader],
            criterion: _Loss,
            optimizer: Optimizer,
            device: torch.device,
            config: Config
    ) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = config.EPOCHS

        self.model = self.model.to(self.device)

    def train_epoch(
            self,
            epoch: int
    ) -> Tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        correct_guesses = 0
        total_samples = 0

        train_loader = self.dataloaders['train']

        for batches_processed, (padded_texts, labels, lengths) in enumerate(train_loader, 1):
            padded_texts, labels = padded_texts.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(padded_texts, lengths)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            exact_predictions = (predictions > 0.0).float()
            correct_guesses += (exact_predictions == labels).sum().item()
            total_samples += labels.size(0)

            if batches_processed % 100 == 0:
                print(f"Epoch {epoch} | Processed {batches_processed}/{len(train_loader)} batches")

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_guesses / total_samples
        return avg_loss, accuracy

    def evaluate(
            self,
            phase: str
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_guesses = 0
        total_samples = 0

        loader = self.dataloaders[phase]

        with torch.no_grad():
            for padded_texts, labels, lengths in loader:
                padded_texts, labels = padded_texts.to(self.device), labels.to(self.device)
                predictions = self.model(padded_texts, lengths)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                exact_predictions = (predictions > 0.0).float()
                correct_guesses += (exact_predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct_guesses / total_samples
        return avg_loss, accuracy

    def train(self) -> None:
        print("Starting training...")
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"Epoch {epoch} finished | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            val_loss, val_acc = self.evaluate('valid')
            print(f"Epoch {epoch} Validation | Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}\n")

    def test(self) -> None:
        print("Testing model...")
        test_loss, test_acc = self.evaluate('test')
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


def main() -> None:
    device = utils.get_device()
    config = Config()

    print("Preparing data...")
    dataloaders, vocab_size = prepare_data(config)

    model = LstmModel(
        num_embeddings=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
    )

    criterion = BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        config.LEARNING_RATE
    )

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
    )

    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
