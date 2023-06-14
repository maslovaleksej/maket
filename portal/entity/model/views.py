from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms
from portal.entity.model.orm import INS
import tensorflow as tf


class MForm(forms.Form):
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
    pass


def model_add_resume(request):
    form = MForm(request.POST)

    if form.is_valid():
        dir_path_present = False

        if (dir_path_present):
            model_orm = INS(
                form
            )
            model_orm.save()
            return redirect(models_list)
        else:
            error = "нет такой папки"
            context = {
                "form": form,
                "error_msg": error
            }

    else:
        error = "не заполнены некототорые поля"

    context = {
        "form": form,
        "error_msg": error
    }
    return render(request, 'portal/pages/model/add_form.html', context)


def model_add_form(request):
    m = MForm()
    context = {"form": m}
    return render(request, 'portal/pages/model/add_form.html', context)


# return render(request, 'portal/pages/model/add.html')


def model_del(request, id):
    INS.objects.filter(id=id).delete()
    return redirect(models_list)
