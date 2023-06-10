from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms
from portal.entity.dataset.orm import Dataset


class DSForm(forms.Form):
    shortName = forms.CharField()
    dir_path = forms.CharField()


@login_required
def dataset_list(request):
    datasets = Dataset.objects.all()
    context = {'datasets': datasets}
    return render(request, 'portal/pages/dataset/list.html', context)


def dataset_detail(request, id):
    dataset = Dataset.objects.filter(pk=id).first()
    context = {'datasets': dataset}
    return render(request, 'portal/pages/dataset/detail.html', context)


def dataset_add(request):
    if request.method == "POST":
        ds = DSForm(request.POST)
        if ds.is_valid():
            shortName = ds.cleaned_data['shortName']
            dir_path = ds.cleaned_data['dir_path']

            dataset = Dataset(shortName=shortName, dir_path=dir_path)
            dataset.class_nums = 10
            dataset.size = 2000
            dataset.save()

            return redirect(dataset_list)

    else:
        ds = DSForm()
    # new = Dataset(shortName=one.name)
    # context = {'datasets': one}
    context = {"form": ds}
    return render(request, 'portal/pages/dataset/add.html', context)


def dataset_del(request, id):
    return redirect(dataset_list)
