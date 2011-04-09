# Author: Arun Chaganty <arunchaganty@gmail.com>
#

# $Id$
FILE = report
OUTPUT = $(shell basename $(PWD))
FIGURES =
SECTIONS = intro.tex 

all: $(OUTPUT).pdf

$(OUTPUT).pdf: $(FILE).tex ${FIGURES} $(SECTIONS) 
	texify < $< > $(FILE)_tmp.tex 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE)_tmp.tex 
	bibtex $(FILE)_tmp
	pdflatex --file-line-error --interaction=nonstopmode $(FILE)_tmp.tex 
	mv $(FILE)_tmp.pdf $@

$(FILE).pdf: $(FILE).tex ${FIGURES} 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE).tex 

${FIGURES}: %.pdf : %.eps 
	epstopdf --autorotate=All $^

.PHONY: clean

clean:
	rm -rf *.{aux,dvi,out,bbl,blg,brf,log}
	rm -rf $(FILE)_tmp.tex

