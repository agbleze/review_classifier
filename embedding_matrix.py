import numpy as np
import torch


class EmbeddingMatrixMaker(object):
    def __init__(self, glove_filepath, words):
        self.glove_filepath = glove_filepath
        self.words = words
        
        
    def load_glove_from_file(self):
        """Load the glove embeddings
        
        Args:
            glove_filepath (str): path to the glove embedding file
            
        Returns:
            word_to_idx (dict): embeddings (numpy.ndarray)
        
        """
        self.word_to_index = {}
        self.embeddings = []
        
        with open(self.glove_filepath) as fp:
            for index, line in enumerate(fp):
                line = line.split(" ") # each line word num1 num2 ...
                self.word_to_index[line[0]] = index
                embedding_i = np.array([float(val) for val in line[1:]])
                self.embeddings.append(embedding_i)
        return self.word_to_index, np.stack(self.embeddings)
        
    def make_embedding_matrix(self):
        """create embedding matrix from a specific set of words
        
        Args:
            glove_fliepath (str): fle path to the glove embeddings
            words (list): list of words in the dataset
        
        """        
        word_to_idx, glove_embeddings = self.load_glove_from_file()
        embedding_size = glove_embeddings.shape[1]
        
        self.final_embeddings = np.zeros((len(self.words), embedding_size))
        
        for i, word in enumerate(self.words):
            if word in word_to_idx:
                self.final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
            else:
                embedding_i = torch.ones(1, embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i)
                self.final_embeddings[i, :] = embedding_i
                
        return self.final_embeddings
    
    
    def save_embedding(self, filename: str = 'embeddings'):
        """save the embedding matrix to a numpy file
        
        Args:
            filename (str): name of the file to save the embedding matrix
        
        """
        np.save(filename, self.final_embeddings)
        
        
    def load_embedding(self, filename: str = 'embeddings.npy'):
        """load the embedding matrix from a numpy file
        
        Args:
            filename (str): name of the file to load the embedding matrix
            
        Returns:
            np.ndarray: embedding matrix
        
        """
        return np.load(filename)
         
 





