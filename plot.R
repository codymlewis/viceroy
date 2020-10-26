#!/usr/bin/env Rscript

library(ggplot2)
library(jsonlite)
library(stringr)


main <- function() {
        options <- fromJSON("options.json")
        attack <- `if`(options$adversaries$type == "onoff",
                sprintf(
                        "On-Off Attack with %d Epoch Toggle",
                        options$adversaries$toggle_time
                ),
                sprintf(
                        "%s Attack",
                        str_to_title(options$adversaries$type)
                )
        )
        title <- sprintf(
                "Performance of %s under %d%% %d->%d %s",
                str_to_title(options$fit_fun),
                options$adversaries$percent_adv * 100,
                options$adversaries$from,
                options$adversaries$to,
                attack
        )
        df <- read.csv('results.log')
        df$epoch <- 1:length(df$accuracy)

        ggplot(data=df, aes(x=epoch)) +
                geom_line(aes(y=accuracy, colour="Accuracy")) +
                geom_line(aes(y=attack_success, colour="Attack Success Rate")) +
                labs(
                     title=title,
                     x="Epochs",
                     y="Rate",
                     colour=NULL
                )  +
                scale_y_continuous(lim=c(0,1)) +
                theme(legend.position="bottom")
        ggsave("plot.png")
}

main()
