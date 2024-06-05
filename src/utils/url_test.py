from urllib import request
from urllib.request import Request, urlopen

url = 'https://huggingface.co/api/models/meta-llama/Meta-Llama-3-8B-Instruct/revision/main'

request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
webpage = urlopen(request_site).read()