from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path, include


urlpatterns = [

    path('',   include('portal.entity.dataset.urls')),
    path('datasets/',   include('portal.entity.dataset.urls')),
    path('models/',     include('portal.entity.model.urls')),


    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='/'), name='logout'),
]

