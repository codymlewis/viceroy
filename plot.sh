#!/bin/sh

Rscript plot.R && \
        mogrify -comment "$(cat options.json)" plot.png
