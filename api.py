
#%%
import joblib
from flask import Flask, request
from flask_restful import Resource, Api
from helpers import predict_category

from classifier import NewsClassifier
from review_dataset import NewsDataset
from torch.nn import Module
import torch
import json
from helpers import (args, load_glove_from_file, 
                     set_seed_everywhere, handle_dirs,
                     compute_accuracy, make_train_state,
                     generate_batches, update_train_state, 
                     predict_category, preprocess_text
                     )
from vectorizer import NewsVectorizer
import re
import pandas as pd
from embedding_matrix import EmbeddingMatrixMaker



file = open(args.vectorizer_file)
vectorizer_file = json.load(file)

review_vocab = vectorizer_file['title_vocab']['token_to_idx'].keys()


review_outcome = vectorizer_file['category_vocab']['token_to_idx']  

#%%

embedding_matrix = EmbeddingMatrixMaker(glove_filepath=args.glove_filepath,
                                        words=review_vocab
                                    )

embeddings = embedding_matrix.load_embedding()


classifier = NewsClassifier(embedding_size=args.embedding_size,
                            num_embeddings=len(review_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim,
                            num_classes=len(review_outcome),
                            dropout_p=args.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0
                            )



classifier.load_state_dict(torch.load(args.model_state_file))

classifier = classifier.to(args.device)

dataset = NewsDataset.load_dataset_and_make_vectorizer(args.data_csv)

vectorizer = dataset.get_vectorizer()

#%%

class RecommendPredictor(Resource):
    @staticmethod
    def post():
        review = request.get_json()['review']
        
        result = predict_category(title=review, classifier=classifier,
                         vectorizer=vectorizer, 
                         max_length=dataset._max_seq_length + 1
                         )  
        
        return result
    
    

app = Flask(__name__)
api = Api(app)


api.add_resource(RecommendPredictor, '/predict')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    
    
    