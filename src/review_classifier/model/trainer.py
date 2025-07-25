import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook, tqdm
import argparse
from typing import Literal
import json

from review_classifier.utils.helpers import (load_glove_from_file, 
                                            set_seed_everywhere, handle_dirs,
                                            #make_embedding_matrix, 
                                            compute_accuracy, make_train_state,
                                            generate_batches, update_train_state
                                            )
from review_classifier.data.review_dataset import ReviewDataset
from review_classifier.model.classifier import ReviewClassifier
from review_classifier.preprocess.vectorizer import ReviewVectorizer
from review_classifier.utils.embedding_matrix import EmbeddingMatrixMaker
from review_classifier.utils.helpers import predict_category, get_samples
from review_classifier import logger
import torch
import os

def train(model, num_epochs: int,
          loss_func, train_state: dict, dataset, optimizer,
          scheduler,
          update_train_state_func=update_train_state,
          evaluate_func=compute_accuracy, batch_size=32,
          early_stopping_criteria=5,
          tqdm_in_cli=True,
          ):
    if not tqdm_in_cli:
        train_bar = tqdm_notebook(desc='split=train',
                            total=dataset.get_num_batches(batch_size), 
                            position=1, 
                            leave=True
                            )
        val_bar = tqdm_notebook(desc='split=val',
                                total=dataset.get_num_batches(batch_size), 
                                position=1, 
                                leave=True
                                )
        epoch_bar = tqdm_notebook(desc='training routine',
                                total=num_epochs,
                                position=0
                                )
    else:
        train_bar = tqdm(desc='split=train',
                            total=dataset.get_num_batches(batch_size), 
                            position=1, 
                            leave=True
                            )
        val_bar = tqdm(desc='split=val',
                        total=dataset.get_num_batches(batch_size), 
                        position=1, 
                        leave=True
                        )
        epoch_bar = tqdm(desc='training routine',
                        total=num_epochs,
                        position=0
                        )
    try:
        for epoch_index in range(num_epochs):
            train_state['epoch_index'] = epoch_index

            running_loss = 0.0
            running_acc = 0.0

            model, running_loss, running_acc, mode_bar = compute_model_performance(model=model,
                                        mode="train",
                                        dataset=dataset,
                                        loss_func=loss_func,
                                        evaluate_func=evaluate_func,
                                        optimizer=optimizer,
                                        mode_bar=train_bar,
                                        epoch_index=epoch_index,
                                        batch_size=batch_size,
                                        #mode_state=train_state
                                        )

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            running_loss = 0.
            running_acc = 0.

            model, running_loss, running_acc, mode_bar = compute_model_performance(model=model,
                                        mode="val",
                                        dataset=dataset,
                                        loss_func=loss_func,
                                        evaluate_func=evaluate_func,
                                        mode_bar=val_bar,
                                        epoch_index=epoch_index,
                                        batch_size=batch_size,
                                        #mode_state=train_state
                                        )

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state_func(early_stopping_criteria, 
                                                  model=model,
                                                  train_state=train_state
                                                  )

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
        return model, train_state
    except KeyboardInterrupt:
        logger.info("User interrupt... Stopping training.")


def compute_model_performance(model, mode: Literal["train", "val", "test"],
                              dataset: ReviewDataset,
                              batch_size,
                              loss_func, epoch_index: int,
                              evaluate_func=compute_accuracy,
                              optimizer: optim.Optimizer = None,
                              mode_bar=None, 
                              device="cuda",
                              ):
                                  
    if mode == "train":
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training mode")
        dataset.set_split('train')
        model.train()
    elif mode == "eval":
        dataset.set_split('val')
        model.eval()
    elif mode == "test":
        dataset.set_split('test')
        model.eval()
    batch_generator = generate_batches(dataset=dataset,
                                       batch_size=batch_size,
                                       device=device,
                                       
                                       )
    running_loss = 0.
    running_acc = 0.

    for batch_index, batch_dict in enumerate(batch_generator):
        if mode == "train":
            # step 1. zero the gradients
            optimizer.zero_grad()
            
        y_pred =  model(batch_dict['x_data'])
        
        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = evaluate_func(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        if mode == "train":
            loss.backward()
            optimizer.step()
        if mode_bar is not None:
            mode_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                epoch=epoch_index
                                )
            mode_bar.update()
    return model, running_loss, running_acc, mode_bar


def model_diagnostics(model, data: dict, vectorizer: ReviewVectorizer, 
                      max_seq_length: int #= dataset._max_seq_length + 1
                      ) -> None:
    for truth, sample_group in data.items():
        print(f"Groundtruth: {truth}")
        print("="*30)
        for sample in sample_group:
            prediction = predict_category(sample, model, 
                                            vectorizer, 
                                            max_seq_length
                                        )
            print("Prediction: {} (p={:0.2f})".format(prediction['category'],
                                                    prediction['probability']))
            print("\t + Sample: {}".format(sample))
        print("-"*30 + "\n")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN for product reviews classification")
    parser.add_argument('--data_csv', type=str,
                        default='data_splitted/review_df_split2.csv',
                        help='Path to the CSV file containing the reviews dataset')
    parser.add_argument('--vectorizer_file', type=str, default='model_store/vectorizer.json',)

    parser.add_argument("--model_state_file", 
                        type=str,
                        help="Path to save the model state file",
                        )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--glove_filepath", type=str,
                        )
    parser.add_argument("--use_glove", action="store_true",
                        required=False
                        )
    parser.add_argument("--embedding_size",
                        type=int,
                        default=100,
                        )
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=100
                        )
    parser.add_argument("--num_channels",
                        type=int,
                        default=100,
                        )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--batch_size",
                        type=int, default=128
                        )
    parser.add_argument("--num_epochs",
                        type=int, default=300, 
                        )
    parser.add_argument("--early_stopping_criteria",
                        default=5,
                        help="Threshold to stop training early if the validation loss does not improve for this many epochs",
                        type=int
                        )
    parser.add_argument("--catch_keyboard_interrupt", action="store_true",
                        help="Catch keyboard interrupt to stop training gracefully"
                        )
    parser.add_argument("--reload_from_files", action="store_true")
    parser.add_argument("--expand_filepaths_to_save_dir", action="store_true",)
    parser.add_argument("--max_seq_length", default=1646, type=int,)
    parser.add_argument("--resume_train", action="store_true",
                        help="Resume training from the last checkpoint. Will use the model_state_file to load the model state and continue training."
                        )
    parser.add_argument("--tqdm_in_cli", action="store_true",
                        help="Use tqdm in CLI mode instead of notebook mode. This is useful for running the script in a terminal or command line interface."
                        )
    parser.add_argument("--diagnose_model", action="store_true",
                        help="Run model diagnostics after training. This will print the predictions for a few samples from the test set."
                        )
    parser.add_argument("--num_samples_to_diagnose", type=int, default=5,
                        help="Number of samples to diagnose after training. This will print the predictions for a few samples from the test set."
                        )
        
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file
                                            )

        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file
                                            )
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    if not torch.cuda.is_available():
        cuda = False
    else:
        cuda = True
        
    device = torch.device("cuda" if cuda else "cpu")
    print("Using CUDA: {}".format(cuda))
    print(f"Using device: {device}")
    # Set seed for reproducibility
    set_seed_everywhere(args.seed, cuda)

    # handle dirs
    handle_dirs(args.save_dir)

    if args.reload_from_files:
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.data_csv,
                                                            args.vectorizer_file
                                                            )
    else:
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.data_csv)
        
        dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    #%% Use GloVe or randomly initialized embeddings
    if args.use_glove:
        logger.info(f"Using GloVe embeddings from {args.glove_filepath}")
        words = vectorizer.review_vocab._token_to_idx.keys()
        embedding_matrix = EmbeddingMatrixMaker(glove_filepath=args.glove_filepath,
                            words=words
                            )
        embeddings = embedding_matrix.make_embedding_matrix()
        embedding_matrix.save_embedding()
        print("Using pre-trained embeddings")
    else:
        logger.info("Using randomly initialized embeddings")
        embeddings = None
    classifier = ReviewClassifier(embedding_size=args.embedding_size,
                                  num_embeddings=len(vectorizer.review_vocab),
                                  num_channels=args.num_channels,
                                  hidden_dim=args.hidden_dim,
                                  num_classes=len(vectorizer.category_vocab),
                                  dropout_p=args.dropout_p,
                                  pretrained_embeddings=embeddings,
                                  padding_idx=0, device=device
                                  )
    if args.resume_train:
        logger.info(f"Resuming training from {args.model_state_file}")
        classifier.load_state_dict(torch.load(args.model_state_file))      
            
    classifier = classifier.to(device)
    dataset.class_weights = dataset.class_weights.to(device)
        
    loss_func = nn.CrossEntropyLoss(dataset.class_weights).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min', factor=0.5,
                                                    patience=1
                                                    )

    train_state = make_train_state(learning_rate=args.learning_rate,
                                   model_state_file=args.model_state_file,
                                   early_stopping_step=args.early_stopping_criteria
                                   )
    model, train_state = train(model=classifier,
                               num_epochs=args.num_epochs,
                               loss_func=loss_func,
                               train_state=train_state,
                               dataset=dataset, 
                               optimizer=optimizer,
                               scheduler=scheduler,
                               update_train_state_func=update_train_state,
                               evaluate_func=compute_accuracy,
                               batch_size=args.batch_size,
                               early_stopping_criteria=args.early_stopping_criteria,
                               tqdm_in_cli=args.tqdm_in_cli
                               )
    if args.diagnose_model:
        logger.info("Running model diagnostics on the test set")
        data = get_samples(dataset=dataset, split_type='test',
                            num_samples=args.num_samples_to_diagnose
                            )
        model_diagnostics(model=model, data=data, vectorizer=vectorizer, 
                        max_seq_length=dataset._max_seq_length + 2
                        ) 
    
    
    logger.info("Evaluating model on the test set")
    model, running_loss, running_acc, _ = compute_model_performance(model=classifier,
                                                                    mode="test",
                                                                    dataset=dataset,
                                                                    loss_func=loss_func,
                                                                    evaluate_func=compute_accuracy,
                                                                    batch_size=args.batch_size,
                                                                    epoch_index=None,
                                                                    mode_bar=None
                                                                    )

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc
        
    logger.info(f"Test loss: {train_state['test_loss']}")
    logger.info(f"Test Accuracy: {train_state['test_acc']}")

    with open("train_state.json", "w") as f:
        logger.info(f"Saving train state to {f.name}")
        json.dump(train_state, f, indent=4)
        
    logger.info(f"Training complete. Model state saved to {args.model_state_file}")
    logger.info(f"Train state saved to train_state.json")

