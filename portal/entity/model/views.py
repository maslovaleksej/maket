from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django import forms
from portal.entity.model.orm import INS
import tensorflow as tf


class MForm(forms.Form):
    type = forms.IntegerField(label="type")
    shortName = forms.CharField(label="Название датасета")
    dir_name = forms.CharField(label="Название корневой папки на диске")
    comments = forms.CharField(label="Описание модели")
    input_size_x = forms.IntegerField()
    input_size_y = forms.IntegerField()
    input_size_ch = forms.IntegerField()
    min = forms.IntegerField()
    max = forms.IntegerField()
    build_shape = forms.BooleanField()
    arguments = forms.JSONField()


class MForm_resume(forms.Form):
    shortName = forms.CharField()
    dirName = forms.CharField()
    comments = forms.CharField()


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
    # form check before add in database
    model_form = MForm(request.POST)
    if model_form.is_valid():
        cleaned_model_form = model_form.cleaned_data

        model_path = "/Users/aleksejmaslov/Data/Classification/Models/Src/efficientnet_b0_feature-vector_1"
        ins = tf.keras.models.load_model(model_path, compile=True)
        # ins.compile()
        summary = ins.summary()
        # print(summary)
        pass

        cleaned_model_form['summary'] = summary[-100:]

        error_msg = None

        context = {
            "form": cleaned_model_form,
            "error_msg": error_msg
        }
        return render(request, 'portal/pages/model/add_resume.html', context)


def model_add_form(request):
    m = MForm()
    context = {"form": m}
    return render(request, 'portal/pages/model/add_form.html', context)


# return render(request, 'portal/pages/model/add.html')


def model_del(request, id):
    INS.objects.filter(id=id).delete()
    return redirect(models_list)
