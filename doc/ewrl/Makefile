# Author: Arun Chaganty <arunchaganty@gmail.com>
#

# $Id$
FILE = report
OUTPUT = $(shell basename $(PWD))
FIGURES = 
FIGURES_CUSTOM = figures/rooms-options.pdf
SECTIONS = macros.tex abstract.tex intro.tex theory.tex experiments.tex conclusions.tex\
		   taxi.tex

all: $(OUTPUT).pdf

$(OUTPUT).pdf: $(FILE).tex ${FIGURES} ${FIGURES_CUSTOM} $(SECTIONS) 
	./texify < $< > $(FILE)_tmp.tex 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE)_tmp.tex 
	bibtex $(FILE)_tmp
	pdflatex --file-line-error --interaction=nonstopmode $(FILE)_tmp.tex 
	mv $(FILE)_tmp.pdf $@

$(FILE).pdf: $(FILE).tex ${FIGURES} 
	pdflatex --file-line-error --interaction=nonstopmode $(FILE).tex 

${FIGURES}: %.pdf : %.eps 
	epstopdf --autorotate=All $^

figures/rooms-options.pdf: rooms.tex
	pdflatex -shell-escape $^;
	mv $(^:.tex=-figure0.pdf) $@;
	rm -f $(^:.tex=-)*
	rm -f $(^:.tex=.auxlock)

.PHONY: clean

clean:
	rm -f *.{aux,bbl,blg,brf,out,log}
	rm -f $(FILE)_tmp.*

