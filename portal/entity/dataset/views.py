from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms

from chain import dataset_util, const
from chain.disk_util import get_datasets_path
from portal.entity.dataset.orm import Dataset
import os


class DSForm(forms.Form):
    shortName = forms.CharField(label="Название датасета")
    dir_name = forms.CharField(label="Название корневой папки на диске")
    comments = forms.CharField(label="Название корневой папки на диске")


class DSForm_full(forms.Form):
    shortName = forms.CharField()
    dirName = forms.CharField()
    comments = forms.CharField()
    size = forms.CharField()
    classNums = forms.CharField()


# ----------------------------------------------------ФАЫ

@login_required
def dataset_list(request):
    datasets = Dataset.objects.all()
    context = {'datasets': datasets}
    return render(request, 'portal/pages/dataset/list.html', context)


def dataset_detail(request, id):
    dataset = Dataset.objects.filter(pk=id).first()
    context = {
        'pk': dataset.pk,
        'shortName': dataset.shortName,
        'dir_name': dataset.dir_name,
        "comments": dataset.comments,
        'size': dataset.size,
        'class_nums': dataset.class_nums
    }

    dir_path = get_datasets_path(dataset.dir_name)
    demo_set = []

    if os.path.isdir(dir_path):
        for class_name in os.listdir(dir_path)[:100]:
            path = os.path.join(dir_path, class_name)

            if os.path.isdir(path):
                files = os.listdir(path)
                demo_set.append([class_name, files[:8]])

    context = {
        'pk': dataset.pk,
        'shortName': dataset.shortName,
        'dir_name': dataset.dir_name,
        "comments": dataset.comments,
        'size': dataset.size,
        'class_nums': dataset.class_nums,
        'demo_set': demo_set
    }


    return render(request, 'portal/pages/dataset/detail.html', context)


def dataset_add(request):
    if request.method == "POST":

        # form check before add in database
        ds = DSForm(request.POST)
        if ds.is_valid():
            shortName = ds.cleaned_data['shortName']
            dir_name = ds.cleaned_data['dir_name']
            comments = ds.cleaned_data['comments']

            dir_path = get_datasets_path(dir_name)

            dg = dataset_util.ImageGenerator(
                dir_path=dir_path
            )

            error_msg = None
            if dg.total_size == 0 or dg.class_nums == 0: error_msg = "Количество изображений = 0"

            dataset = Dataset.objects.filter(shortName=shortName).first()
            if dataset: error_msg = "Имя датасета присутствует в дазе данных "

            dataset = Dataset.objects.filter(dir_name=dir_name).first()
            if dataset: error_msg += " Данная директория уже подключена к базе данных"

            context = {"shortName": shortName,
                       "dir_name": dir_name,
                       "comments": comments,
                       "size": dg.total_size,
                       "class_nums": dg.class_nums,
                       "class_names": dg.class_names,
                       "error_msg": error_msg
                       }
            return render(request, 'portal/pages/dataset/add_resume.html', context)

        # add in database
        ds_full = DSForm_full(request.POST)
        if ds_full.is_valid():
            dataset_orm = Dataset(
                shortName=ds_full.cleaned_data['shortName'],
                dir_name=ds_full.cleaned_data['dirName'],
                class_nums=ds_full.cleaned_data['classNums'],
                size=ds_full.cleaned_data['size'],
            )
            dataset_orm.save()
            return redirect(dataset_list)

    if request.method == "GET":
        ds = DSForm()
        context = {"form": ds}
        return render(request, 'portal/pages/dataset/add.html', context)


def dataset_del(request, id):
    Dataset.objects.filter(id=id).delete()
    return redirect(dataset_list)
