
from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.form_view),
    path('form_view', views.form_view, name='form_view')
]
