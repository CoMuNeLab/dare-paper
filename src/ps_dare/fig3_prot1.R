library(ggplot2)
library(dplyr)
library(ggrepel)
library(tidyr)
library(reticulate)

# The fitted models were created in the project's pmdarima environment.
use_condaenv("ps-dare-paper", required = TRUE)

extract_model_pvalues <- function(models) {
    pickle <- import("pickle")
    builtins <- import_builtins()
    numpy <- import("numpy", convert = FALSE)

    extract_one <- function(model_file) {
        stream <- builtins$open(model_file, "rb")
        model <- tryCatch(
            pickle$load(stream),
            finally = stream$close()
        )

        # pmdarima ARIMA objects expose the fitted statsmodels result here.
        result <- model$arima_res_
        parameter_names <- as.character(py_to_r(result$param_names))
        estimates <- as.numeric(py_to_r(numpy$asarray(result$params)))
        pvalues <- as.numeric(py_to_r(numpy$asarray(result$pvalues)))

        tibble(
            parameter = as.character(parameter_names),
            estimate = estimates,
            pvalue = pvalues
        )
    }

    bind_rows(lapply(seq_len(nrow(models)), function(i) {
        model_row <- models[i, , drop = FALSE]
        extracted <- extract_one(model_row$model_file[[1]])

        bind_cols(
            model_row %>%
                select(
                    model_id, dataset_name, testing_end, driver_kind,
                    driver, lag, model_file
                ) %>%
                slice(rep(1, nrow(extracted))),
            extracted
        )
    }))
}

results <- read.csv("/Users/tommasobertola/Git/ps-dare-paper-2/data/output/protocol-1/res_prot1.csv")
results %>% colnames()

dataset_labels <- c(
    general_giustiniani = "General - ER Giustiniani",
    general_osa = "General - ER Sant'Antonio",
    general_pediatrico = "General - ER Pediatrico",
    hw_giustiniani = "HRI - ER Giustiniani",
    hw_osa = "HRI - ER Sant'Antonio",
    hw_pediatrico = "HRI - ER Pediatrico",
    ili_giustiniani = "ILI - ER Giustiniani",
    ili_osa = "ILI - ER Sant'Antonio",
    ili_pediatrico = "ILI - ER Pediatrico"
)

driver_kind_labels <- c(
    none = "No driver",
    # forced = "Forced",
    # pollution = "Pollution",
    weather = "Weather"
)

driver_labels <- c(
    abs_hu = "Absolute humidity",
    CO_mean = "CO (mean)",
    CO_peak = "CO (peak)",
    frost = "Frost",
    heating_degree_days = "Heating degree days",
    hu = "Humidity",
    humidex = "Humidex",
    indoor_wet_bulb = "Indoor wet-bulb temperature",
    NO2_mean = "NO2 (mean)",
    NO2_peak = "NO2 (peak)",
    none = "No driver",
    pm10_mean = "PM10 (mean)",
    pm10_peak = "PM10 (peak)",
    `pm2.5_mean` = "PM2.5 (mean)",
    `pm2.5_peak` = "PM2.5 (peak)",
    SO2_mean = "SO2 (mean)",
    SO2_peak = "SO2 (peak)",
    summer_days = "Summer days",
    tropical_night = "Tropical night"
)

lag_labels <- c(
    `0` = "0",
    `0-1` = "0-1",
    `1` = "1"
)

prot_1_weekly <- results %>%
    filter(model_file != "") %>%
    filter(temporal_aggregation == "weekly") %>%
    # A facet is a single dataset, so only plot one row for each model
    # configuration represented in that panel.
    distinct(dataset_name, lag, subset, driver, .keep_all = TRUE) %>%
    group_by(dataset_name) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    ungroup()


prot_1_weekly_baseline <- results %>%
    filter(model_file == "") %>%
    left_join(
        prot_1_weekly %>%
            group_by(dataset_name) %>%
            summarise(
                point_min = min(mape),
                point_max = max(mape), .groups = "drop"
            ),
        by = "dataset_name"
    ) %>%
    filter(!is.na(point_min), !is.na(point_max)) %>%
    filter(temporal_aggregation == "weekly") %>%
    mutate(
        blue_max = if_else(mape >= point_max, Inf, mape),
        red_min = if_else(mape <= point_min, -Inf, mape)
    )

baseline_labels_2025 <- setNames(
    sprintf(
        "%s\nBaseline MAPE: %.2f%%",
        dataset_labels[prot_1_weekly_baseline$dataset_name],
        prot_1_weekly_baseline$mape
    ),
    prot_1_weekly_baseline$dataset_name
)


model_pvalues <- bind_rows(
    extract_model_pvalues(prot_1_weekly),
) %>%
    mutate(
        is_driver_parameter = !grepl(
            "^(ar\\.|ma\\.|intercept$|sigma2$)",
            parameter
        ),
        is_significant = is_driver_parameter & pvalue < 0.05
    ) %>%
    arrange(testing_end, dataset_name, model_id, parameter)

significant_models <- model_pvalues %>%
    group_by(model_file) %>%
    summarise(significant = any(is_significant, na.rm = TRUE), .groups = "drop")

prot_1_weekly <- bind_rows(
    prot_1_weekly %>% filter(driver_kind != "none"),
    prot_1_weekly %>%
        filter(driver_kind == "none") %>%
        distinct(dataset_name, aic, mape, driver_kind, .keep_all = TRUE)
) %>%
    left_join(significant_models, by = "model_file") %>%
    mutate(significant = coalesce(significant, FALSE))

ggplot() +
    geom_rect(
        data = prot_1_weekly_baseline %>% filter(mape > point_min),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = blue_max),
        fill = "lightblue",
        alpha = 0.35,
        inherit.aes = FALSE
    ) +
    geom_rect(
        data = prot_1_weekly_baseline %>% filter(mape < point_max),
        aes(xmin = -Inf, xmax = Inf, ymin = red_min, ymax = Inf),
        fill = "lightcoral",
        alpha = 0.25,
        inherit.aes = FALSE
    ) +
    geom_point(
        data = prot_1_weekly %>% filter(driver_kind == "none"),
        aes(x = delta_aic, y = mape, shape='none'),
        size = 2.5,
        color = "black",
    ) +
    geom_point(
        data = prot_1_weekly %>% filter(driver_kind != "none"),
        mapping = aes(x = delta_aic, y = mape, color = lag, shape = driver_kind),
        size = 2.5
    ) +
    geom_point(
        data = prot_1_weekly %>% filter(significant),
        mapping = aes(x = delta_aic, y = mape),
        shape = 21,
        fill = NA,
        color = "black",
        size = 4,
        stroke = 0.8,
        inherit.aes = FALSE
    ) +
    geom_text_repel(
        data = prot_1_weekly %>%
            group_by(dataset_name) %>%
            slice_min(mape, n = 1, with_ties = FALSE) %>%
            ungroup(),
        mapping = aes(
            x = delta_aic,
            y = mape,
            label = unname(driver_labels[driver])
        ),
        # fontface = "bold",
        color = "black",
        size = 3,
        nudge_x = 4,
        hjust = 0,
        direction = "y",
        box.padding = 1,
        point.padding = 0.3,
        min.segment.length = 0,
        segment.color = "black",
        segment.linewidth = 0.4,
        segment.alpha = 1,
        show.legend = FALSE
    ) +
    facet_wrap(
        ~dataset_name,
        scales = "free",
        labeller = ggplot2::as_labeller(baseline_labels_2025)
    ) +
    scale_color_discrete(
        name = "Lag (weeks)",
        breaks = c("0", "0-1", "1"),
        labels = lag_labels,
        guide = guide_legend(order = 2)
    ) +
    scale_shape_manual(
        name = "Driver type",
        values = c(none = 18, weather = 16),
        breaks = c("none", "weather"),
        labels = driver_kind_labels,
        guide = guide_legend(order = 1)
    ) +
    scale_y_continuous(labels = scales::label_percent(scale = 1)) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom") +
    labs(
        x = expression(Delta * "AIC"),
        y = "MAPE"
    )


ggsave(
    filename = "graphs/fig3_prot1_weekly.pdf",
    width = 12,
    height = 8.5,
    units = "in",
)




prot_1_weekly %>%
    group_by(dataset_name) %>%
    filter(mape == min(mape) | driver_kind == "none") %>%
    select(dataset_name, lag, driver_kind, driver, aic, mape, ae) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    arrange(dataset_name, lag, driver_kind) %>%
    write.csv("graphs/fig3_prot1_weekly.csv", row.names = FALSE)


prot_1_monthly <- results %>%
    filter(model_file != "") %>%
    filter(temporal_aggregation == "monthly") %>%
    # A facet is a single dataset, so only plot one row for each model
    # configuration represented in that panel.
    distinct(dataset_name, lag, subset, driver, .keep_all = TRUE) %>%
    group_by(dataset_name) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    ungroup()


prot_1_monthly_baseline <- results %>%
    filter(model_file == "") %>%
    left_join(
        prot_1_monthly %>%
            group_by(dataset_name) %>%
            summarise(
                point_min = min(mape),
                point_max = max(mape), .groups = "drop"
            ),
        by = "dataset_name"
    ) %>%
    filter(!is.na(point_min), !is.na(point_max)) %>%
    filter(temporal_aggregation == "monthly") %>%
    mutate(
        blue_max = if_else(mape >= point_max, Inf, mape),
        red_min = if_else(mape <= point_min, -Inf, mape)
    )

baseline_labels_monthly <- setNames(
    sprintf(
        "%s\nBaseline MAPE: %.2f%%",
        dataset_labels[prot_1_monthly_baseline$dataset_name],
        prot_1_monthly_baseline$mape
    ),
    prot_1_monthly_baseline$dataset_name
)


model_pvalues_monthly <- extract_model_pvalues(prot_1_monthly) %>%
    mutate(
        is_driver_parameter = !grepl(
            "^(ar\\.|ma\\.|intercept$|sigma2$)",
            parameter
        ),
        is_significant = is_driver_parameter & pvalue < 0.05
    ) %>%
    arrange(testing_end, dataset_name, model_id, parameter)

significant_models_monthly <- model_pvalues_monthly %>%
    group_by(model_file) %>%
    summarise(significant = any(is_significant, na.rm = TRUE), .groups = "drop")

prot_1_monthly <- bind_rows(
    prot_1_monthly %>% filter(driver_kind != "none"),
    prot_1_monthly %>%
        filter(driver_kind == "none") %>%
        distinct(dataset_name, aic, mape, driver_kind, .keep_all = TRUE)
) %>%
    left_join(significant_models_monthly, by = "model_file") %>%
    mutate(significant = coalesce(significant, FALSE))

ggplot() +
    geom_rect(
        data = prot_1_monthly_baseline %>% filter(mape > point_min),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = blue_max),
        fill = "lightblue",
        alpha = 0.35,
        inherit.aes = FALSE
    ) +
    geom_rect(
        data = prot_1_monthly_baseline %>% filter(mape < point_max),
        aes(xmin = -Inf, xmax = Inf, ymin = red_min, ymax = Inf),
        fill = "lightcoral",
        alpha = 0.25,
        inherit.aes = FALSE
    ) +
    geom_point(
        data = prot_1_monthly %>% filter(driver_kind == "none"),
        aes(x = delta_aic, y = mape, shape='none'),
        size = 2.5,
        color = "black",
    ) +
    geom_point(
        data = prot_1_monthly %>% filter(driver_kind != "none"),
        mapping = aes(x = delta_aic, y = mape, color = lag, shape = driver_kind),
        size = 2.5
    ) +
    geom_point(
        data = prot_1_monthly %>% filter(significant),
        mapping = aes(x = delta_aic, y = mape),
        shape = 21,
        fill = NA,
        color = "black",
        size = 4,
        stroke = 0.8,
        inherit.aes = FALSE
    ) +
    geom_text_repel(
        data = prot_1_monthly %>%
            group_by(dataset_name) %>%
            slice_min(mape, n = 1, with_ties = FALSE) %>%
            ungroup(),
        mapping = aes(
            x = delta_aic,
            y = mape,
            label = unname(driver_labels[driver])
        ),
        # fontface = "bold",
        color = "black",
        size = 3,
        nudge_x = 4,
        hjust = 0,
        direction = "y",
        box.padding = 1,
        point.padding = 0.3,
        min.segment.length = 0,
        segment.color = "black",
        segment.linewidth = 0.4,
        segment.alpha = 1,
        show.legend = FALSE
    ) +
    facet_wrap(
        ~dataset_name,
        scales = "free",
        labeller = ggplot2::as_labeller(baseline_labels_monthly)
    ) +
    scale_color_discrete(
        name = "Lag (months)",
        breaks = c("0", "0-1", "1"),
        labels = lag_labels,
        guide = guide_legend(order = 2)
    ) +
    scale_shape_manual(
        name = "Driver type",
        values = c(none = 18, weather = 16),
        breaks = c("none", "weather"),
        labels = driver_kind_labels,
        guide = guide_legend(order = 1)
    ) +
    scale_y_continuous(labels = scales::label_percent(scale = 1)) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom") +
    labs(
        x = expression(Delta * "AIC"),
        y = "MAPE"
    )


ggsave(
    filename = "graphs/fig3_prot1_monthly.pdf",
    width = 12,
    height = 8.5,
    units = "in",
)


prot_1_monthly %>%
    group_by(dataset_name) %>%
    filter(mape == min(mape) | driver_kind == "none") %>%
    select(dataset_name, lag, driver_kind, driver, aic, mape, ae) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    arrange(dataset_name, lag, driver_kind) %>%
    write.csv("graphs/fig3_prot1_monthly.csv", row.names = FALSE)
