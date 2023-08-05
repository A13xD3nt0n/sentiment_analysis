from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect

from comment.classification.predict import get_prediction


@csrf_protect
def comment(request):
    if request.method == 'POST':
        text = request.POST.get('text', None)
        if text is None:
            return render(request, 'comment.html',
                          {'result': False})
        else:
            predict_sentiment, predict_mark = get_prediction(text)
            return render(request, 'comment.html',
                          {'result': True, 'predict_mark': predict_mark, 'predict_sentiment': predict_sentiment})
    else:
        return render(request, 'comment.html',
                      {'result': False})


if __name__ == '__main__':
    print(get_prediction('The film is awesome'))
