from decouple import config

HOST = config('HOST')
PORT = config('PORT')
ENDPOINT = config('ENDPOINT')


URL = f'{HOST}:{PORT}{ENDPOINT}'
