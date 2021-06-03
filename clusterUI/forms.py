from django import forms

class ParameterForm(forms.Form):
    sample = forms.IntegerField(label='sample',required=False)
    threshold = forms.IntegerField(label='threshold',required=False)
    kval = forms.IntegerField(label='kval',required=False)
    recheck = forms.IntegerField(label='recheck',required=False)