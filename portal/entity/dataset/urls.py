from django.urls import path
from portal.entity.dataset.views import *

urlpatterns = [
    path('', dataset_list, name='dataset_list'),
    path('detail/<int:id>', dataset_detail, name='dataset_detail'),
    path('add', dataset_add, name='dataset_add'),
    path('del/<int:id>', dataset_del, name='dataset_del'),
]
