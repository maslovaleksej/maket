from django.urls import path
from portal.entity.experiment.views import *

urlpatterns = [
    path('', experiments_list, name='experiments_list'),
    path('detail/<int:id>', experiment_detail, name='experiment_detail'),

    path('add_form', experiment_add_form, name='experiment_add_form'),
    path('add_resume', experiment_add_resume, name='experiment_add_resume'),

    path('del/<int:id>', experiment_del, name='experiment_del'),
]
