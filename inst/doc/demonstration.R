## -----------------------------------------------------------------------------
library(fastei)

eim_apo <- get_eim_chile(elect_district = "APOQUINDO")

eim_apo

## -----------------------------------------------------------------------------
eim_apo <- run_em(eim_apo)
eim_apo$prob

## -----------------------------------------------------------------------------
eim_apo <- bootstrap(eim_apo, seed = 42, nboot = 30)
eim_apo$sd

## -----------------------------------------------------------------------------
eim_nav <- get_eim_chile(elect_district = "NAVIDAD")
eim_nav <- bootstrap(eim_nav, seed = 42, nboot = 30)
eim_nav$sd

## ----message=FALSE, warning=FALSE, echo=TRUE----------------------------------
library(ggplot2)
library(reshape2)
library(viridis)

plot_district <- function(matrix1, district1, matrix2, district2, sd = FALSE) {
    value <- ifelse(sd == FALSE, "prob", "sd")
    df1 <- melt(matrix1)
    df2 <- melt(matrix2)
    df1$Matrix <- district1
    df2$Matrix <- district2
    combined_df <- rbind(df1, df2)
    color <- ifelse(value == "prob", "plasma", "viridis")

    # Add text to each cell of the matrix
    combined_df$label <- sprintf("%.2f", combined_df$value)
    combined_df$text_color <- ifelse(combined_df$value > round(max(combined_df$value) * 0.75 + min(combined_df$value) * 0.25, 2), "black", "white")
    districts <- sort(c(district1, district2))

    # Call the plot
    ggplot(combined_df, aes(x = Var2, y = Var1, fill = value)) +
        geom_tile() +
        geom_text(aes(label = label, color = text_color), size = 3) +
        scale_fill_viridis(
            name = value,
            option = color,
        ) +
        scale_color_identity() +
        facet_wrap(~Matrix) +
        coord_fixed() +
        theme_bw() +
        labs(
            title = ifelse(value == "prob",
                paste("Estimated probabilities in districts:", districts[1], "and", districts[2]),
                paste("Standard deviation of estimated probabilities in districts:", districts[1], "and", districts[2])
            ),
            x = "Candidates' votes", y = "Voters' age range", fill = value
        )
}

## ----navidad_apoq_sd_comparison, fig.width = 8, fig.height = 6.5, fig.cap = "Navidad and Apoquindo standard deviation comparison", fig.align = "center", message=FALSE, warning=FALSE, results="hide"----
plot_district(
    matrix1 = eim_nav$sd, district1 = "Navidad",
    matrix2 = eim_apo$sd, district2 = "Apoquindo", sd = TRUE
)

## -----------------------------------------------------------------------------
eim_nav_proxy <- get_agg_proxy(eim_nav, seed = 42, sd_threshold = 0.1, sd_statistic = "average")
eim_nav_proxy$group_agg

## -----------------------------------------------------------------------------
mean(eim_nav$sd) - mean(eim_nav_proxy$sd)

## -----------------------------------------------------------------------------
plot_matrix <- function(mat, sd = FALSE, y_labels = NULL) {
    # Initial configurations
    if (!sd) mat <- t(mat)
    df <- reshape2::melt(mat)
    colnames(df) <- c("Row", "Column", "Value")
    df$Row <- factor(df$Row, levels = rev(sort(unique(df$Row))))
    df$Column <- factor(df$Column, levels = sort(unique(df$Column)))
    if (!sd) {
        df$Label <- sprintf("%d", df$Value)
        title_text <- "Voters distribution"
        x_lab <- "Ballot Box"
        y_lab <- "Dem. Group"
        fill_lab <- "Voters"
        df$text_color <- ifelse(df$Value > 30, "black", "white")
        option <- "inferno"
        start <- 0.5
        limits <- NULL
    } else {
        df$Label <- sprintf("%.2f", df$Value)
        title_text <- "Standard deviation of estimated probabilities on district: Navidad"
        x_lab <- "Candidates' votes"
        y_lab <- "Voters' age range"
        fill_lab <- "sd"
        df$text_color <- ifelse(df$Value > 0.13, "black", "white")
        option <- "viridis"
        start <- 0
        limits <- c(0, 0.2)
    }

    # Plot
    p <- ggplot(df, aes(x = Column, y = Row, fill = Value)) +
        geom_tile() +
        geom_text(aes(label = Label, color = text_color), size = 3) +
        scale_color_identity() +
        scale_fill_viridis_c(option = option, begin = start, limits = limits) +
        coord_fixed() +
        theme_bw() +
        theme(axis.text.y = element_text(size = 7), axis.text.x = element_text(size = 7)) +
        labs(
            title = title_text,
            x = x_lab,
            y = y_lab,
            fill = fill_lab
        )
    # Add custom y-axis labels if provided
    if (!is.null(y_labels)) {
        p <- p + scale_y_discrete(labels = y_labels)
    }
    p
}

## ----standard_deviation_proxy, fig.width = 8, fig.height = 3.5, fig.cap = "Navidad aggregated standard deviation with proxy method", fig.align = "center", message=FALSE, warning=FALSE, echo=TRUE----
plot_matrix(eim_nav_proxy$sd, sd = TRUE, y_labels = c("X18.49", "X50."))

## -----------------------------------------------------------------------------
eim_nav_opt <- get_agg_opt(eim_nav, seed = 42, sd_threshold = 0.1, sd_statistic = "average")
eim_nav_opt$group_agg

## ----standard_deviation_opt, fig.width = 8, fig.height = 3.5, fig.cap = "Navidad aggregated standard deviation with opt method", fig.align = "center", message=FALSE, warning=FALSE----
plot_matrix(eim_nav_opt$sd, sd = TRUE, y_labels = c("X18.49", "X50.69", "X70."))

## -----------------------------------------------------------------------------
eim_prov <- get_eim_chile("PROVIDENCIA")
eim_prov <- run_em(eim_prov)
eim_apo <- run_em(eim_apo)

## ----prov_apoc_comparison, fig.width = 8, fig.height = 6.5, fig.cap = "Providencia and Apoquindo comparison", fig.align = "center", message=FALSE, warning=FALSE, echo=TRUE----
plot_district(eim_apo$prob, "Apoquindo", eim_prov$prob, "Providencia")

## -----------------------------------------------------------------------------
comparison <- welchtest(
    object1 = eim_prov,
    object2 = eim_apo,
    method = "mult",
    nboot = 30,
    seed = 42
)

round(comparison$pvals, 3)

## -----------------------------------------------------------------------------
eim_gra <- get_eim_chile("LA GRANJA")
eim_gra <- run_em(eim_gra)
eim_bar <- get_eim_chile("LO BARNECHEA")
eim_bar <- run_em(eim_bar)

## ----granja_lobarnechea_comparison, fig.width = 8, fig.height=6.5, fig.cap = "Lo Barnechea and La Granja comparison", fig.align = "center", message=FALSE, warning=FALSE, echo = FALSE, results="hide"----
plot_district(eim_gra$prob, "La Granja", eim_bar$prob, "Lo Barnechea")

## -----------------------------------------------------------------------------
comparison2 <- welchtest(
    object1 = eim_gra,
    object2 = eim_bar,
    method = "mult",
    nboot = 30,
    seed = 42,
)

round(comparison2$pvals, 3)

## -----------------------------------------------------------------------------
eim_sim <- simulate_election(num_ballots = 15, num_groups = 2, num_candidates = 3, seed = 42)
eim_sim

## -----------------------------------------------------------------------------
eim_sim$real_prob

## -----------------------------------------------------------------------------
eim_sim <- run_em(eim_sim)
eim_sim$prob

## -----------------------------------------------------------------------------
input_probability <- matrix(c(0.9, 0.05, 0.05, 0.2, 0.3, 0.5), nrow = 2, byrow = TRUE)
input_probability

## -----------------------------------------------------------------------------
eim_sim2 <- simulate_election(
    num_ballots = 30, num_groups = 2, num_candidates = 3, seed = 42,
    prob = input_probability
)
eim_sim2
eim_sim2$real_prob

## -----------------------------------------------------------------------------
eim_sim3 <- simulate_election(
    num_ballots = 20, num_groups = 4, num_candidates = 2, seed = 42,
    lambda = 0.1
)

## ----eim_sim3_heatmap, fig.width = 8, fig.height = 3.5, fig.cap = "Voters' heatmap for a low lambda value", fig.align = "center", message=FALSE, warning=FALSE, results="hide"----
plot_matrix(eim_sim3$W)

## -----------------------------------------------------------------------------
run_em(eim_sim3)$prob

## -----------------------------------------------------------------------------
eim_sim4 <- simulate_election(
    num_ballots = 20, num_groups = 4, num_candidates = 2, seed = 42,
    lambda = 0.9
)

## ----eim_sim4_heatmap, fig.width = 8, fig.height = 3.5, fig.cap = "Voters' heatmap for a high lambda value", fig.align = "center", message=FALSE, warning=FALSE, results="hide"----
plot_matrix(eim_sim4$W)

## -----------------------------------------------------------------------------
run_em(eim_sim4)$prob

