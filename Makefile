.PHONY: venv install dev test lint run docker-build docker-run clean

VENV=.venv
PYTHON=$(VENV)/Scripts/python.exe

venv:
	python -m venv $(VENV)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	$(PYTHON) -m pip install -r requirements.txt

dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m voice_conv.cli --ref data/ref/A.wav --src data/src/B.wav --out data/out/C.wav --vc-sr 48000

docker-build:
	cd docker && docker build -t voiceconv .

docker-run:
	cd docker && docker compose up --build

clean:
	rm -rf $(VENV) build dist *.egg-info __pycache__ */__pycache__
