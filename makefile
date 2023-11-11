install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

scene1:
	python3 scenario1.py

scene2:
	python3 scenario2.py

scene3:
	python3 scenario3.py

cleanpics:
	rm *png

clean:
	rm -rf venv
	find -iname "*.pyc" -delete
