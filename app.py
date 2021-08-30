from flask import Flask
import validate

app = Flask(__name__)

@app.route('/')
def hello_world():
    str: str = """
    <h2> docker_ssd is running </h2>
    Call <a href='./validate'>validate</a>
    <br><br><br>
    <img src='.\static\docker_001.png'/>"
    """
    return str


@app.route('/validate')
def run_validate():
    #
    validate.run()
    # get files names
    str = """
    <h1>OK</h1>
     
    <img src='.\static\docker_001.png'/>
    """
    return str


if __name__ == "__main__":
    app.run(host='0.0.0.0')
