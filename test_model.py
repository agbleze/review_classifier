
#%%
from classifier import NewsClassifier
from review_dataset import NewsDataset
from torch.nn import Module
import torch
import json
from helpers import (args, load_glove_from_file, 
                     set_seed_everywhere, handle_dirs,
                     #make_embedding_matrix, 
                     compute_accuracy, make_train_state,
                     generate_batches, update_train_state, 
                     predict_category, preprocess_text
                     )
from vectorizer import NewsVectorizer
import re
import pandas as pd
from embedding_matrix import EmbeddingMatrixMaker


# %%
file = open(args.vectorizer_file)
vectorizer = json.load(file)

#%%

#words = vectorizer.title_vocab._token_to_idx.keys()

review_vocab = vectorizer['title_vocab']['token_to_idx'].keys()


#%%

embedding_matrix = EmbeddingMatrixMaker(glove_filepath=args.glove_filepath,
                                        words=review_vocab
                                    )

embeddings = embedding_matrix.load_embedding()


#%%   
train_state = make_train_state(args)    
 
#%%

len(review_vocab)

review_outcome = vectorizer['category_vocab']['token_to_idx']  

len(review_outcome)  

#%%
classifier = NewsClassifier(embedding_size=args.embedding_size,
                            num_embeddings=len(review_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim,
                            num_classes=len(review_outcome),
                            dropout_p=args.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0
                            )

#classifier.load_state_dict(torch.load(model_path))

#%%
classifier.load_state_dict(torch.load(args.model_state_file))

classifier = classifier.to(args.device)




# %%
vectorizer.keys()
# %%


len(vectorizer['title_vocab'])


vectorizer['title_vocab']


        
#%% Inference
# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(pattern=r"([.,!?])", repl=r" \1 ", string=text)
    text = re.sub(pattern=r"[^a-zA-Z.,!?]+", repl=r" ", string=text)       
    return text


#%%
def predict_category(title, classifier, vectorizer, max_length):
    """Predicts a news category for a new title
    
    Args:
        title (str): a raw title string
        classifier (NewsVectorizer): an instanve of the trained classifier
        vectorizer (NewsVectorizer): the corresponding vectorizer
        max_length (int): the max sequence length
            CNN are sensitive to the input data tensor size, 
            This ensures to keep it the same size as the training data
    """
    title = preprocess_text(title)
    vectorized_title = torch.tensor(vectorizer.vectorize(title, vector_length=max_length))
    result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    predicted_category = vectorizer.category_vocab.lookup_index(indices.item())
    
    return {'category': predicted_category,
            'probability': probability_values.item()
            }
    
    
 
 #%%   

# training from a checkpoint
    
    
#dataset = NewsDataset(news_df= pd.read_csv(args.data_csv))    
# dataset = NewsDataset.load_dataset_and_load_vectorizer(args.data_csv,
#                                                        args.vectorizer_file
#                                                     )

#%%

dataset = NewsDataset.load_dataset_and_make_vectorizer(args.data_csv)

#%%
vectorizer = dataset.get_vectorizer()



#%%


predict_category("I hate the product", classifier, vectorizer, dataset._max_seq_length)





# %%

df_lower = pd.read_csv('data_splitted/review_df_split.csv')

# %%

df_lower.columns


#%%

#col_lower = lambda x: x.lower()

# %%
df_lower['reviews.doRecommend'] = [x.lower() for x in df_lower['reviews.doRecommend']]



# %%
df_str = df_lower.astype({'reviews.doRecommend': 'str'}, )


#%%

small_letter = lambda x: x.lower()

#%%

df_str['reviews.doRecommend'] = df_str['reviews.doRecommend'].apply(small_letter)


#%%

df_str.to_csv('data_splitted/review_df_split2.csv')

