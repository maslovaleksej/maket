from django.urls import path

from portal.entity.model.views import *


urlpatterns = [
    path('', models_list, name='models_list'),
    path('detail/<int:id>', model_detail, name='model_detail'),
    path('add', model_add, name='model_add'),
    path('del/<int:id>', model_del, name='model_del'),
]
