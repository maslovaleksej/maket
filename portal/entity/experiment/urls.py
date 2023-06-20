from django.urls import path
from portal.entity.experiment.views import *

urlpatterns = [
    path('', experiments_list, name='experiments_list'),
    path('detail/<int:id>', experiment_detail, name='experiment_detail'),
    path('status/<int:id>', experiment_status, name='experiment_status'),
    path('result/<int:id>', experiment_result, name='experiment_result'),
    path('run/<int:id>', experiment_run, name='experiment_run'),

    path('add_form', experiment_add_form, name='experiment_add_form'),
    path('add_resume', experiment_add_resume, name='experiment_add_resume'),

    path('del/<int:id>', experiment_del, name='experiment_del'),
]
