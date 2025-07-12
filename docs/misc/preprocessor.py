
#%%
from review_classifier.data.review_dataset import ReviewDataset
import pandas as pd

#%%
raw_datapath = ReviewDataset.get_datapath(data_foldername='data', data_filename='product_reviews.csv')


#%%
data_segmented = ReviewDataset.segment_dataset(data_path=raw_datapath)


def trim_dataset(data: pd.DataFrame):
    """Select only required features and removes reviews from unverified sources

    Args:
        data (pd.DataFrame): _description_
    """

    review_df =  data[['reviews.text', 'reviews.doRecommend', 'split']]
    review_df_clean = review_df[review_df['reviews.text']!='Rating provided by a verified purchaser'].dropna()
    return review_df_clean


review_df_cleaned = trim_dataset(data=data_segmented)

review_df_cleaned.to_csv(ReviewDataset.get_datapath(data_foldername='data_splitted', data_filename='review_df_split.csv'))








# %%
