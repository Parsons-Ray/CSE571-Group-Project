cse571_report.pdf: cse571_report.tex ./ref.bib 
	pdflatex -shell-escape cse571_report < /dev/null
	bibtex cse571_report
	pdflatex -shell-escape cse571_report < /dev/null
	pdflatex -shell-escape cse571_report < /dev/null

clean:
	rm -rf _minted-cse571_report *.lof *.toc *.dvi *.bbl *.blg *.aux *.log *.out x.tex *.pyg
