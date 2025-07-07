import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook
import argparse

from review_classifier.utils.helpers import (args, load_glove_from_file, 
                                            set_seed_everywhere, handle_dirs,
                                            #make_embedding_matrix, 
                                            compute_accuracy, make_train_state,
                                            generate_batches, update_train_state
                                            )
from review_classifier.data.review_dataset import ReviewDataset
from review_classifier.model.classifier import ReviewClassifier
from review_classifier.preprocess.vectorizer import ReviewVectorizer
from review_classifier.utils.embedding_matrix import EmbeddingMatrixMaker
from review_classifier.utils.helpers import predict_category
#%%
parser = argparse.ArgumentParser(description="Train a CNN for product reviews classification")
parser.add_argument('--data_csv', type=str, default='data_splitted/review_df_split2.csv',
                    help='Path to the CSV file containing the reviews dataset')
parser.add_argument('--vectorizer_file', type=str, default='model_store/vectorizer.json',
if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
    
# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
    
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)


#%% Initializations
args.use_glove = True

if args.reload_from_files:
    # training from a checkpoint
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
    words = vectorizer.title_vocab._token_to_idx.keys()
    embedding_matrix = EmbeddingMatrixMaker(glove_filepath=args.glove_filepath,
                         words=words
                        )
    embeddings = embedding_matrix.make_embedding_matrix()
    embedding_matrix.save_embedding()
    # embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
    #                                    words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = True
    

#%%
classifier = ReviewClassifier(embedding_size=args.embedding_size,
                            num_embeddings=len(vectorizer.title_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim,
                            num_classes=len(vectorizer.category_vocab),
                            dropout_p=args.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0
                            )
            
        
        

#%% Training loop
classifier = classifier.to(args.device)
dataset.class_weights = dataset.class_weights.to(args.device)
    
loss_func = nn.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)

train_state = make_train_state(args)

epoch_bar = tqdm_notebook(desc='training routine', 
                          total=args.num_epochs,
                          position=0)

dataset.set_split('train')
train_bar = tqdm_notebook(desc='split=train',
                          total=dataset.get_num_batches(args.batch_size), 
                          position=1, 
                          leave=True)
dataset.set_split('val')
val_bar = tqdm_notebook(desc='split=val',
                        total=dataset.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

try:
    for epoch_index in range(args.num_epochs):
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: batch generator, set loss and acc to 0, set train mode on

        dataset.set_split('train')
        batch_generator = generate_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = classifier(batch_dict['x_data'])

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  epoch=epoch_index)
            train_bar.update()

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset

        # setup: batch generator, set loss and acc to 0; set eval mode on
        dataset.set_split('val')
        batch_generator = generate_batches(dataset, 
                                           batch_size=args.batch_size, 
                                           device=args.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred =  classifier(batch_dict['x_data'])

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                            epoch=epoch_index)
            val_bar.update()

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=args, model=classifier,
                                         train_state=train_state)

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            break

        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.update()
except KeyboardInterrupt:
    print("Exiting loop")



#%% TESTING 

# compute the loss & accuracy on the test set using the best available model

classifier.load_state_dict(torch.load(train_state['model_filename']))

classifier = classifier.to(args.device)
dataset.class_weights = dataset.class_weights.to(args.device)
loss_func = nn.CrossEntropyLoss(dataset.class_weights)

dataset.set_split('test')
batch_generator = generate_batches(dataset, 
                                   batch_size=args.batch_size, 
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred =  classifier(batch_dict['x_data'])
    
    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'])
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

    
print("Test loss: {};".format(train_state['test_loss']))
print("Test Accuracy: {}".format(train_state['test_acc']))



#%% Inference
# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(pattern=r"([.,!?])", repl=r" \1 ", string=text)
    text = re.sub(pattern=r"[^a-zA-Z.,!?]+", repl=r" ", string=text)       
    return text


#%%
# def predict_category(title, classifier, vectorizer, max_length):
#     """Predicts a news category for a new title
    
#     Args:
#         title (str): a raw title string
#         classifier (NewsVectorizer): an instanve of the trained classifier
#         vectorizer (NewsVectorizer): the corresponding vectorizer
#         max_length (int): the max sequence length
#             CNN are sensitive to the input data tensor size, 
#             This ensures to keep it the same size as the training data
#     """
#     title = preprocess_text(title)
#     vectorized_title = torch.tensor(vectorizer.vectorize(title, vector_length=max_length))
#     result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
#     probability_values, indices = result.max(dim=1)
#     predicted_category = vectorizer.category_vocab.lookup_index(indices.item())
    
#     return {'category': predicted_category,
#             'probability': probability_values.item()
#             }


#%%
def get_samples():
    samples = {}
    for cat in dataset.val_df['reviews.doRecommend'].unique():
        samples[cat] = dataset.val_df['reviews.text'][dataset.val_df['reviews.doRecommend']==cat].tolist()[:5]
    return samples

val_samples = get_samples()


#%%
#title = input("Enter a news title to classify: ")
classifier = classifier.to("cpu")

def model_diagnostics(model, data: dict, vectorizer: ReviewVectorizer, 
                      max_seq_length: int = dataset._max_seq_length + 1
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
    


#%%
predict_category("I the best product", classifier, vectorizer, dataset._max_seq_length +1)




# %%
