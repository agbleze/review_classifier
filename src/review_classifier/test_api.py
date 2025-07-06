
#%%

from api_utils import request_prediction

from constant import (HOST, PORT, ENDPOINT)

URL = f'{HOST}:{PORT}{ENDPOINT}'

data = "this is a GREAT PROUDCT BUT I DON'T LIKE IT"


request_prediction(URL = URL, review_data = data)



# %%
