from django.urls import path

from portal.entity.model.views import *

urlpatterns = [
    path('', models_list, name='models_list'),
    path('detail/<int:id>', model_detail, name='model_detail'),

    path('add_form', model_add_form, name='model_add_form'),
    path('add_resume', model_add_resume, name='model_add_resume'),
    path('add', model_add, name='model_add'),

    path('del/<int:id>', model_del, name='model_del'),
]
