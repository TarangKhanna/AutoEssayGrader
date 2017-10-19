from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.login, name='login'),
	url(r'^signup/$',views.signup, name='signup'),
	url(r'^upload_essay/$',views.upload_essay, name='upload_essay'),
	url(r'^view_past_essays/$',views.view_past_essays, name='view_past_essays'),
	url(r'^contact_us/$',views.contact_us, name='contact_us'),
	url(r'^logout/$',views.logout, name='logout'),

	url(r'^submit_essay/$',views.submit_essay, name='submit_essay'),

]