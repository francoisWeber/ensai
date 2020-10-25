.ONESHELL:

.PHONY: install
install:
	virtualenv env
	env/bin/pip3.8 install -r reqs.txt

jupyter:
	env/bin/python -m ipykernel install --user --name=Ensai

clean:
	rm -r TP1/outputs