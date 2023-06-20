import os

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms

from chain.disk_util import get_src_models_path, get_dir_size
from portal.entity.dataset.orm import Dataset
from portal.entity.experiment.orm import Experiment
from portal.entity.model.orm import INS

import subprocess


class EForm(forms.Form):
    type = forms.IntegerField()
    shortName = forms.CharField()
    dir_name = forms.CharField()
    comments = forms.CharField(required=False)
    input_size_x = forms.IntegerField()
    input_size_y = forms.IntegerField()
    input_size_ch = forms.IntegerField()
    min = forms.IntegerField()
    max = forms.IntegerField()
    batch_norm_momentum = forms.CharField(required=False)


# ----------------------------------------------------

@login_required
def experiments_list(request):
    experiments = Experiment.objects.order_by('dir_name').all()
    context = {'experiments': experiments}
    return render(request, 'portal/pages/experiment/list.html', context)


def experiment_add_form(request):

    datasets = Dataset.objects.all().order_by('shortName')
    models = INS.objects.all().order_by('shortName')
    e = EForm()

    context = {
        "form": e,
        "models": models,
        "datasets": datasets
    }
    return render(request, 'portal/pages/experiment/add_form.html', context)


def experiment_detail(request, id):
    form = Experiment.objects.filter(pk=id).first()
    context = {
        "form": form
    }
    return render(request, 'portal/pages/experiment/detail.html', context)

def experiment_status(request, id):
    form = Experiment.objects.filter(pk=id).first()
    context = {
        "form": form
    }
    return render(request, 'portal/pages/experiment/status.html', context)

def experiment_result(request, id):
    form = Experiment.objects.filter(pk=id).first()
    context = {
        "form": form
    }
    return render(request, 'portal/pages/experiment/result.html', context)

def experiment_add_resume(request):
    form = EForm(request.POST)
    error_msg = ''

    if form.is_valid():
        model = INS.objects.filter(shortName=form.cleaned_data['shortName']).first()
        if model: error_msg = "Имя модели присутствует в дазе данных"

        model = INS.objects.filter(dir_name=form.cleaned_data['dir_name']).first()
        if model: error_msg = "Данная модель (папка с моделью) уже подключена к базе данных"


        dir_path = get_src_models_path(form.cleaned_data['dir_name'])
        if not os.path.isdir(dir_path): error_msg = 'нет такой папки на диске'

        dir_size = get_dir_size(dir_path)
        if dir_size == 0: error_msg = 'размер папки равен 0'

        try:
            batch_norm_momentum = float(form.cleaned_data['batch_norm_momentum'])
        except:
            batch_norm_momentum = None

        if len(error_msg) == 0:
            model_orm = INS(
                type=form.cleaned_data['type'],
                dir_name=form.cleaned_data['dir_name'],
                shortName=form.cleaned_data['shortName'],
                comments=form.cleaned_data['comments'],
                input_size_x=form.cleaned_data['input_size_x'],
                input_size_y=form.cleaned_data['input_size_y'],
                input_size_ch=form.cleaned_data['input_size_ch'],
                min=form.cleaned_data['min'],
                max=form.cleaned_data['max'],
                batch_norm_momentum=batch_norm_momentum,
                dir_size=dir_size,
            )
            model_orm.save()
            return redirect(experiments_list)

    else:
        error_msg = "не заполнены некототорые поля"

    context = {
        "form": form,
        "error_msg": error_msg
    }
    return render(request, 'portal/pages/experiment/add_form.html', context)



def experiment_del(request, id):
    INS.objects.filter(id=id).delete()
    return redirect(experiments_list)


def experiment_run(request, id):
    result = subprocess.run(['python3', '-V'], encoding='utf-8')
    return redirect(experiments_list)
