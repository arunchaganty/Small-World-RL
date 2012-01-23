# Author: Arun Chaganty <arunchaganty@gmail.com>
#

# $Id$
FILE=presentation
OUTPUT=$(shell basename $(PWD))
FIGURES=

$(OUTPUT).pdf: $(FILE).tex ${FIGURES}
	pdflatex --file-line-error --interaction=nonstopmode $<
	mv $(FILE).pdf $@

${FIGURES}: %.pdf : %.eps 
	epstopdf --autorotate=All $^

.PHONY: clean

clean:
	rm -rf $(FILE).{aux,dvi,out,log,nav,snm,toc}

