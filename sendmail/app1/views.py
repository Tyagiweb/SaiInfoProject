#from django.shortcuts import render

# Create your views here.

# views.py
from django.shortcuts import render
from django.core.mail import send_mail

def send_email(request):
    if request.method == 'POST':
        subject = request.POST['subject']
        message = request.POST['message']
        from_email = 'prashanttyagipt17@gmail.com'
        recipient_list = [request.POST['email']]

        try:
            send_mail(subject, message, from_email, recipient_list)
            success_message = 'Email sent successfully!'
        except Exception as e:
            error_message = f'Email could not be sent. Error: {str(e)}'

    return render(request, 'EmailFrom.html', locals())


def mail_name(request):
    return render(request,'Name.html')





