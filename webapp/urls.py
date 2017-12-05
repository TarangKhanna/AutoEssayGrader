from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.login, name='login'),
	url(r'^signup/$',views.signup, name='signup'),
	url(r'^upload_essay/$',views.upload_essay, name='upload_essay'),
	url(r'^view_past_essays/$',views.view_past_essays, name='view_past_essays'),
	url(r'^batch_grading/$',views.batch_process_essays, name='batch_grading'),
	url(r'^logout/$',views.logout, name='logout'),

	url(r'^submit_essay/$',views.submit_essay, name='submit_essay'),
	url(r'^submit_essay_batch/$',views.submit_essay_batch, name='submit_essay_batch'),
]