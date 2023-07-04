cd venv/Scripts
py -m venv env
set FLASK_APP=../../main.py
set FLASK_ENV=developement
set PYTHONUNBUFFERED=1;FLASK_APP=main.py;FLASK_DEBUG=1;LANG=EN_EN.UTF-8;FLASK_RUN_PORT=8888
flask run --port=8888