from django.shortcuts import render
from django.http import HttpResponseNotFound
from .ta import *
import numpy as np

def index(request):
    if request.POST:
        code = request.POST.get('code')
        period = request.POST.get('period')
        interval = request.POST.get('interval')
        willr = int(request.POST.get('willr'))
        bias = int(request.POST.get('bias'))
        rsi = int(request.POST.get('rsi'))
        k_9 = 9
        raw_df = get_raw_df(code, period, interval)
        if raw_df.empty:
            # if the stock was not in the yahoo finance database
            return HttpResponseNotFound("This stock is not in the yahoo finance database")
        else:
            isempty = check_empty(raw_df, willr, bias, rsi)
            if isempty:
                # if any ta index was empty
                return HttpResponseNotFound("You need to set longer period or shorter interval to calculate ta.")
            else:
                Divergences = get_plot(raw_df, willr, bias, rsi)
                return render(request, 'index.html', {'Divergences':Divergences}) 
    else:
        return render(request, 'index.html', {}) 