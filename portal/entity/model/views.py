from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def models(request):
    context = {'group_arr': 'group_arr'}
    return render(request, 'portal/pages/models/list.html', context)

def models_add(request):
    context = {'group_arr': 'group_arr'}
    return render(request, 'portal/pages/models/add.html', context)
