# Author: Arun Chaganty <arunchaganty@gmail.com>
#

# $Id$
FILE = report
OUTPUT = $(shell basename $(PWD))
FIGURES =
SECTIONS = macros.tex abstract.tex intro.tex theory.tex experiments.tex conclusions.tex

all: $(OUTPUT).pdf

$(OUTPUT).pdf: $(FILE).tex ${FIGURES} $(SECTIONS) 
	texify < $< > $(FILE)_tmp.tex 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE)_tmp.tex 
	mv $(FILE)_tmp.pdf $@

$(FILE).pdf: $(FILE).tex ${FIGURES} 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE).tex 

${FIGURES}: %.pdf : %.eps 
	epstopdf --autorotate=All $^

.PHONY: clean

clean:
	rm -f *.{aux,bbl,blg,brf,out,log}
	rm -f $(FILE)_tmp.*

