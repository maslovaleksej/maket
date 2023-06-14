from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms

from chain import dataset_util
from chain.disk_util import get_datasets_path
from portal.entity.dataset.orm import Dataset
import os

from portal.entity.model.orm import INS


class MForm(forms.Form):
    type = forms.Select()
    shortName = forms.CharField(label="Название датасета")
    dir_name = forms.CharField(label="Название корневой папки на диске")
    comments = forms.CharField(label="Описание модели")


class MForm_resume(forms.Form):
    shortName = forms.CharField()
    dirName = forms.CharField()
    comments = forms.CharField()


# ----------------------------------------------------ФАЫ

@login_required
def models_list(request):
    models = INS.objects.all()
    context = {'models': models}
    return render(request, 'portal/pages/model/list.html', context)


def model_detail(request, id):
    model = INS.objects.filter(pk=id).first()
    context = {
        "model": model
    }

    return render(request, 'portal/pages/model/detail.html', context)


def model_add(request):
    if request.method == "POST":
        # form check before add in database
        model_form = MForm(request.POST)
        if model_form.is_valid():
            type = model_form.cleaned_data['type']
            shortName = model_form.cleaned_data['shortName']
            dir_name = model_form.cleaned_data['dir_name']
            comments = model_form.cleaned_data['comments']
            #
            #         dir_path = get_datasets_path(dir_name)
            #
            #         dg = dataset_util.ImageGenerator(
            #             dir_path=dir_path
            #         )
            #
            error_msg = None
            #         if dg.total_size == 0 or dg.class_nums == 0: error_msg = "Количество изображений = 0"
            #
            #         dataset = Dataset.objects.filter(shortName=shortName).first()
            #         if dataset: error_msg = "Имя датасета присутствует в дазе данных "
            #
            #         dataset = Dataset.objects.filter(dir_name=dir_name).first()
            #         if dataset: error_msg += " Данная директория уже подключена к базе данных"
            #
            context = {"shortName": shortName,
                       "dir_name": dir_name,
                       "comments": comments,

                       "error_msg": error_msg
                       }
            return render(request, 'portal/pages/model/add_resume.html', context)
    #
    #     # add in database
    #     ds_full = DSForm_full(request.POST)
    #     if ds_full.is_valid():
    #         dataset_orm = Dataset(
    #             shortName=ds_full.cleaned_data['shortName'],
    #             dir_name=ds_full.cleaned_data['dirName'],
    #             class_nums=ds_full.cleaned_data['classNums'],
    #             size=ds_full.cleaned_data['size'],
    #         )
    #         dataset_orm.save()
    #         return redirect(dataset_list)
    #
    if request.method == "GET":
        m = MForm()
        context = {"form": m}
        return render(request, 'portal/pages/model/add.html', context)


    return render(request, 'portal/pages/model/add.html')


def model_del(request, id):
    INS.objects.filter(id=id).delete()
    return redirect(models_list)
