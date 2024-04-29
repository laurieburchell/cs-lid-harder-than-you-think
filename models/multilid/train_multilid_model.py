""" 
Trains model using pre-generated dataset
usage: python train_multilid_model.py TRAINING_DATASET
"""

import argparse
import torch
from torchtext.data.utils import get_tokenizer
from slowtext import SlowtextClassifier, SlowtextTrainingDataset, SlowtextEvalDataset,\
    build_dataloader, train_loop, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("training_dataset", help="Slowtext training dataset filepath")
parser.add_argument("dev_dataset", help="Slowtext small eval dataset filepath")
parser.add_argument("model_path", help="where to save trained model")
parser.add_argument("--batch_size", help="batch size for training dataloader", type=int, default=256)
parser.add_argument("--device", help="train model on cpu or gpu", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--emb_dim", help="embedding dimension of model", type=int, default=256)
parser.add_argument("--lr", help="learning rate", type=float, default=0.08)
parser.add_argument("--patience", help="patience for reducing learning rate", type=int, default=4)
parser.add_argument("--factor", help="factor with which to multiply lr on stall", type=float, default=0.5)
parser.add_argument("--max_epochs", help="max number of epochs to run", type=int, default=300)
parser.add_argument("--early_stop", help="how many iterations to wait to stop stalled model", type=int, default=15)
args = parser.parse_args()

print("loading training data")
train_dataset = torch.load(args.training_dataset)
train_dataloader_params = {'batch_size': args.batch_size, 'shuffle': True}
train_dataloader = build_dataloader(train_dataset, args.device, **train_dataloader_params)

print("loading evaluation data")
tokeniser = get_tokenizer("moses")
dev_dataset = SlowtextEvalDataset(args.dev_dataset, tokeniser, train_dataset)
dev_dataloader = build_dataloader(dev_dataset, args.device, batch_size=2048)

print("instantiating model")
model = SlowtextClassifier(train_dataset, args.emb_dim).to(args.device)
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, 
                                                       mode='max',
                                                       patience=args.patience, 
                                                       factor=args.factor, 
                                                       verbose=True)

print("starting training loop")
best_f1 = 0
stalled = 0
for epoch in range(args.max_epochs):
    print(f"~~~~ Epoch: {epoch+1} ~~~~")
    train_loop(dataloader=train_dataloader, model=model, 
            loss_fn=loss_fn, optimiser=optimiser, epoch=epoch)
    
    # evaluate at end of epoch
    print(f"evaluating at end of epoch {epoch+1}")
    # get f1 score on small dev set
    steps = (epoch+1) * len(train_dataloader.dataset)
    eval_report_dict = evaluate(dev_dataloader, model, as_dict=True)
    this_f1 = eval_report_dict['macro avg']['f1-score']
    this_lr = optimiser.param_groups[0]['lr']
    print(f"f1 is {this_f1:.3f}, lr is {this_lr:.3f}")

    # check if f1 is improving
    if this_f1 > best_f1:
        stalled = 0
        best_f1 = this_f1
        print(f"New best f1: {this_f1:.3f}! Saving model at {args.model_path}.best_f1")
        torch.save(model.state_dict(), f"{args.model_path}.best_f1")
    else:
        stalled += 1
        print(f"model stalled {stalled} times")
        if stalled > args.early_stop:
            print("halting training")
            break
    scheduler.step(this_f1)
    
print(f"Training complete! Saving model at {args.model_path}")
torch.save(model.state_dict(), args.model_path)