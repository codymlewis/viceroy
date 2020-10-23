#!/usr/bin/env Rscript

library(ggplot2)


main <- function() {
        df <- read.csv('results.log')
        df$epoch <- 1:length(df$accuracy)

        ggplot(data=df, aes(x=epoch)) +
                geom_line(aes(y=accuracy, colour="Accuracy")) +
                geom_line(aes(y=attack_success, colour="Attack rate")) +
                labs(
                     title="Attack Success Rate and Accuracy of FoolsGold under no Attacks",
                     x="Epochs",
                     y="Rate",
                     colour=NULL
                )  +
                scale_y_continuous(lim=c(0,1))
        ggsave("plot.png")
}

main()
