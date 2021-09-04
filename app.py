import os
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
    <h2> docker_ssd v.0.0.5 is running ... </h2>
    Execute <a href='./validate'>validate</a>
    <br><br><br>
    Show <a href='./debug'>debug</a>
    <br><br><br>
    <img src='.\static\docker_001.png'/>
    """
    return str


@app.route('/debug')
def debug():
    str: str = """
    <a href='/'>home</a>
    <h2> Debug Info </h2>
    ...
    
    <h2> History </h2>
    <br><strong>04.02.21</strong> show images as 200x200, sort resulting png-files, fix docker port
    <br><strong>03.02.21</strong> try to fix issue with Create Folder
    <br><strong>03.02.21</strong> add --show result PNG-Files    
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
    log = validate.run()
    ret = """
    <h1>OK</h1>
    """

    # TODO get and present results (generated PNGs)

    folder_res = 'data/MulSet/set20'  # TODO magic string, managed outside of this func
    folder_static = 'static/generated'
    pattern = "*.png"

    #pathlib.Path(folder_static).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(folder_static):
        os.makedirs(folder_static)

    if not os.path.exists(folder_static):
        ret += "<br> folder NOT available" + folder_static

    validate.remove_files(folder_static, pattern)

    files = glob.glob(folder_res + '/' + pattern)
    for f in files:
        #ret += "<br> copy file" + folder_res + '/' + os.path.basename(f)
        shutil.copy(f, folder_static)

    files = glob.glob(folder_static + '/' + pattern)
    sorted_files = sorted(files)

    ret += "<div style='display:flex'>"
    for f in sorted_files:
        ret += "<div style='margin-left:5px;margin-left:5px'>"
        temp = os.path.basename(f)
        ret += "<img src='" + folder_static + "/" + temp + "' width='200' height='200'/>"
        ret += "<div style='margin-left:15px;'>" + temp + "</div>"
        ret += "</div>"
    ret += "</div>"

    ret +="<div>" + log + "</div>"

    return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0')
