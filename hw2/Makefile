.PHONY: all
all: package

.PHONY: package
package: main.py decision_stump.py requirements.txt README.md
	rm -rf r08922176 r08922176.zip;
	mkdir r08922176;
	cp main.py decision_stump.py requirements.txt README.md r08922176;
	zip -r r08922176.zip r08922176;
