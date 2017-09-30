from django.template import RequestContext
from django.shortcuts import render
from essaygrader.urls import navigation_tree

def get_user(request):
    if request != None:
        '''Returns the user stored in the request metadata or None if no user is found'''
        if 'REMOTE_USER' in request.META:
            return request.META['REMOTE_USER'].split('@')[0] # Per documentation, Sentry might include the @ANT.AMAZON.COM
    return None

def render_to_populated_response(page, attributes = {}, request = None):
    template_nav = {'navigation_tree':navigation_tree, 'user_name': get_user(request)}
    #combined_attributes = dict(template_nav.items() + attributes.items())
    combined_attributes = template_nav.copy()
    combined_attributes.update(attributes)
    if request:
        return render(request,page,combined_attributes)
    else:
        return render(page, combined_attributes)

def get_param_with_default(request, param_name, default_value):
    '''Gets param_name from the request.  Returns default_value if param is not present,
    or is empty string

    If the type of the default_value is a boolean then will compare the value to 'TRUE'.
    '''
    if param_name in request.GET:
        value = request.GET[param_name]
        if value is not None and value != '':
            if type(default_value) == bool:
                return value.upper() == 'TRUE'
            return value
    return default_value
