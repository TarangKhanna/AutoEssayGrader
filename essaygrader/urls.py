"""essaygrader URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from collections import namedtuple

NavigationNode = namedtuple('NavigationNode', \
        ['url',\
        'displayName',\
        'children',\
        ])

navigation_tree = [NavigationNode(url='/webapp/upload_essay', displayName='Upload Essay', children=None),\
NavigationNode(url='/webapp/view_past_essays', displayName='View Past Essays', children=None),\
NavigationNode(url='/webapp/batch_grading', displayName='Batch Grading', children=None),\
NavigationNode(url='/webapp/logout', displayName='Log Out', children=None),\
]

urlpatterns = [
    url(r'^admin/', admin.site.urls),\
    url(r'^$', include('webapp.urls')),\
    url(r'^webapp/', include('webapp.urls')),\
    ]