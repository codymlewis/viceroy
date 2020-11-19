#!/usr/bin/env Rscript

library(ggplot2)
library(jsonlite)
library(stringr)
library(itertools)


main <- function(results) {
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
                vals <- c(0)
                attacking <- c()
                toggles <- recycle(options$adversaries$toggle_times)
                while(tail(vals, 1) < length(df$epoch)) {
                        toggle <- nextElem(toggles)
                        vals <- c(vals, tail(vals, 1) + toggle)
                        attacking <- c(attacking, rep(
                                `if`(
                                        length(attacking) == 0,
                                        0,
                                        `if`(tail(attacking, 1) == 1, 0, 1)
                                ),
                                toggle
                        ))
                }
                vals <- c(vals[c(-1, -length(vals))], length(df$epoch))
                start <- df$epoch[attacking == 0]
                end <- df$epoch[attacking == 1]
                ids <- c(1:length(vals) %% 2, `if`(length(vals) %% 2 == 1, 0, NULL))
                rects <- data.frame(
                        start=vals[ids == 1],
                        end=vals[ids == 0],
                        group=rep(c("on", "off"), length(ids) / 2)
                )
                gp <- gp + geom_rect(
                        data=rects,
                        inherit.aes=FALSE,
                        aes(
                            xmin=start,
                            xmax=end,
                            ymin=0,
                            ymax=1,
                        ),
                        color="transparent",
                        fill="orange",
                        alpha=0.3
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
        plot_img <- str_replace_all(
                sprintf(
                        "%s %d %s.png",
                        options$fit_fun,
                        options$adversaries$percent_adv * 100,
                        options$adversaries$type),
                " ",
                "_"
        )
        ggsave(plot_img)
        return(plot_img)
}

RESULTS <- "results.log"
PLOT <- main(RESULTS)
cat(sprintf("Done. Saved plot to %s\n", PLOT))
