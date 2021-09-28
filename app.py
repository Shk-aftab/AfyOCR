from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from utils.yoloRecognition import inference
import os

app = Flask(__name__)

@app.route('/', methods=['POST', "GET"])
@cross_origin()
def main():
    if request.method == 'POST':
        try:
            if request.files.get("img", ""):
                image = request.files.get("img", "")
                if 'static' not in os.listdir():
                    os.mkdir('static')
                image.save('static/image.jpg')
                result_input = 'static/image.jpg'
                result_output = inference(result_input)
                return render_template("index.html", msg=[result_input, result_output])

        except Exception as e:
            print(e)
            result = ('Please pass proper input :' + str(e), 'error')
            return render_template('index.html', msg=result)
    return render_template("index.html")


if __name__ == "__main__":
    # if 'static' not in os.listdir():
    #     os.mkdir('static')
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)