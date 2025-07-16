from django.contrib import admin
from . import views
from django.urls import path, include

urlpatterns = [path("", views.render_home_page),
               path("classification", views.render_classify_page),
               path("classify", views.render_classification),
               path("prediction", views.render_predict_page),
               path("predict", views.render_prediction)]