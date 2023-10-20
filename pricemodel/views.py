from django.shortcuts import render
import numpy as np
import pandas as pd
import urllib.parse
import tensorflow as tf

import os
import joblib
import numpy as np
from django.shortcuts import render
from .forms import URL

import numpy as np
import pandas as pd
import urllib.parse
import tensorflow as tf


def convert_url(raw_url):
    # Tokenize the URL
    tokenized_url = urllib.parse.quote(raw_url)

    # Standardize the URL
    standardized_url = urllib.parse.urlsplit(tokenized_url).geturl()

    # Truncate or pad the URL
    max_url_length = 200  # Maximum length of the padded URL

    if len(standardized_url) > max_url_length:
        # Truncate the URL if it is longer than the maximum length
        truncated_url = standardized_url[-max_url_length:]
        padded_url = [ord(char) for char in truncated_url]
    else:
        # Pad the URL with zeros if it is shorter than the maximum length
        padded_url = [0] * (max_url_length - len(standardized_url)) + [ord(char) for char in standardized_url]
    
    return padded_url

def predict_price(request):
    if request.method == 'POST':
        form = URL(request.POST)
        if form.is_valid():
            # Load the trained linear regression model
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'PRISM_Model.pkl')
            model = joblib.load(model_path)

            # Extract input data from the form
            new_data = np.array(list(form.cleaned_data.values())).reshape(1, -1)

            #url clean
            url_data = form.cleaned_data['url']

            # Perform prediction
            p1 = convert_url(url_data)
            p1 = np.array(p1)

            output = model.predict(p1)
            if output >= 0.5:
                predicted_price = 1
            else:
                predicted_price = 0

            #predicted_price = model.predict(url_data)

            ans=0
            if(predicted_price == 1):
                ans=True
            else:
                ans=False    

            # Prepare the response
            context = {
                'form': form,
                'predicted_price': ans,
            }
            return render(request, 'index.html', context)
    else:
        form = URL()

    context = {'form': form}
    return render(request, 'index.html', context)

'''
raw_url = "https://example.com/path?param1=value1&param2=value2"
padded_url = convert_url(raw_url)
print(padded_url)

p = np.array(padded_url)

output = model.predict(p)

    if output[i] >= 0.5:
        output[i] = 1
    else:
        output[i] = 0

'''