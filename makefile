venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || python3 -m venv venv
	. venv/bin/activate; python3 -m pip install -r requirements.txt
	touch venv/touchfile

# test: venv
#     . venv/bin/activate; nosetests project/test

clean:
	rm -rf venv
	# find -iname "*.pyc" -delete