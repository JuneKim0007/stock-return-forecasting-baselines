.PHONY: pdf clean measure

pdf:
	latexmk -pdf REPORT.tex

clean:
	latexmk -C REPORT.tex
	rm -f REPORT.aux REPORT.bbl REPORT.blg REPORT.fdb_latexmk REPORT.fls REPORT.log REPORT.out REPORT.toc

measure:
	python -m src.runner $(if $(N),--target $(N)) $(if $(SEED),--seed $(SEED))
