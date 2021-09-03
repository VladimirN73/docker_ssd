import os.path

from flask import Flask
import validate
import sys
import pathlib
import glob
import shutil

app = Flask(__name__)

@app.route('/')
def hello_world():
    str: str = """
    <h2> docker_ssd v.0.0.3 is running ... </h2>
    Execute <a href='./validate'>validate</a>
    <br><br><br>
    Show <a href='./debug'>debug</a>
    <br><br><br>
    <img src='.\static\docker_001.png'/>"
    """
    return str


@app.route('/debug')
def debug():
    str: str = """
    <a href='/'>home</a>
    <h2> Debug Info </h2>
    ... [provide debug info] ...
    
    
    <h2> History </h2>
    <h3> v.0.0.3</h3>
    <br><strong>03.02.21</strong> add --show result PNG-Files    
    <br><br>
    ... [provide debug info] ...
    """
    return str

@app.route('/validate')
def run_validate():
    #
    ret = "<a href='/'> home </a>"
    try:
        ret += run_validate_internal()
    except:
        ret += """
        <h1>Error</h1>
        """
        print("Unexpected error:", sys.exc_info()[0])
        ret += str(sys.exc_info()[0]) #TODO encode the string ...

    return ret


def run_validate_internal():
    #
    validate.run()
    ret = """
    <h1>OK</h1>
    """

    # TODO get and present results (generated PNGs)

    folder_res = '.\data\MulSet\set20'  # TODO magic string, managed outside of this func
    folder_static = '.\static\generated'
    pattern = "*.png"

    pathlib.Path(folder_static).mkdir(parents=True, exist_ok=True)

    validate.remove_files(folder_static, pattern)

    files = glob.glob(folder_res + '\\' + pattern)
    for f in files:
        ret += "<br> copy file" + folder_res + '\\' + os.path.basename(f)
        shutil.copy(f, folder_static)

    files = glob.glob(folder_static + '\\' + pattern)
    for f in files:
        temp = os.path.basename(f)
        ret += "<br>" + temp
        ret += "<img src='" + folder_static + "\\" + temp + "'/>"

    return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0')
