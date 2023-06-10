from django.urls import path
from portal.entity.model.views import models, models_add

urlpatterns = [
    path('', models, name='models'),
    path('add/', models_add, name='model_add'),
]
