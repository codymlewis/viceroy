#!/usr/bin/env Rscript

library(ggplot2)
library(jsonlite)
library(stringr)


main <- function(results, plot_img) {
        options <- fromJSON("options.json")
        attack <- `if`(options$adversaries$type == "on off",
                sprintf(
                        "On-Off Attack with %d Epoch Toggle",
                        options$adversaries$toggle_time
                ),
                sprintf(
                        "%s Attack",
                        str_to_title(options$adversaries$type)
                )
        )
        attack <- `if`(options$adversaries$percent_adv > 0,
                sprintf(
                        "%d%% %d->%d %s",
                        options$adversaries$percent_adv * 100,
                        options$adversaries$from,
                        options$adversaries$to,
                        attack
                ),
                "No Attack"
        )
        title <- sprintf(
                "Performance of %s under %s",
                str_to_title(options$fit_fun),
                attack
        )
        df <- read.csv(results)
        df$epoch <- 1:length(df$accuracy)

        gp <- ggplot(data=df, aes(x=epoch))
        if(options$adversaries$percent_adv > 0 &&
           options$adversaries$type == "on off") {
                off <- as.logical(ceiling(df$epoch / options$adversaries$toggle_time) %% 2)
                status <- rep(1, length(df$epoch))
                status[off] <- 0
                start <- df$epoch[status == 0]
                end <- df$epoch[status == 1]
                vals <- seq(1, length(df$epoch), by=options$adversaries$toggle_time)
                vals <- c(vals[-1], length(df$epoch))
                ids <- 1:length(vals) %% 2
                rects <- data.frame(start=vals[ids == 0], end=vals[ids == 1], group=seq_along(start))
                gp <- gp + geom_rect(
                        data=rects,
                        inherit.aes=FALSE,
                        aes(
                                xmin=start,
                                xmax=end,
                                ymin=0,
                                ymax=1,
                                group=group
                        ),
                        color="transparent",
                        fill="orange",
                        alpha=0.01
                )
        }

        gp +
                `if`(options$adversaries$percent_adv > 0,
                        geom_line(
                                aes(
                                    y=attack_success,
                                    colour="Attack Success Rate"
                                )
                        ),
                        NULL
                ) +
                geom_line(aes(y=accuracy, colour="Accuracy")) +
                labs(
                     title=title,
                     x="Epochs",
                     y="Rate",
                     colour=NULL
                )  +
                scale_y_continuous(lim=c(0,1)) +
                theme(
                        legend.position="bottom",
                        plot.title=element_text(size=11)
                )
        ggsave(plot_img)
}

RESULTS <- "results.log"
PLOT <- "plot.png"
cat(sprintf("Plotting results in %s\n", RESULTS))
main(RESULTS, PLOT)
cat(sprintf("Done. Saved plot to %s\n", PLOT))
