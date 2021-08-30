from flask import Flask
import validate
import sys

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
    try:
        validate.run()
        ret = """
        <h1>OK</h1>

        <img src='.\static\docker_001.png'/>
        """

        # TODO get and present results (generated PNGs)

    except:
        ret = """
        <h1>Error</h1>
        """
        print("Unexpected error:", sys.exc_info()[0])
        ret += str(sys.exc_info()[0]) #TODO encode the string ...

    return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0')
