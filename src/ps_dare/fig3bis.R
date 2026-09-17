library(ggplot2)
library(dplyr)
library(ggrepel)


results <- read.csv("/Users/tommasobertola/Git/ps-dare-paper-2/data/weekly/metrics/results5.csv")
results %>% colnames()

dataset_labels <- c(
    general_giustiniani = "General - ER Giustiniani",
    general_osa = "General - ER Sant'Antonio",
    general_pediatrico = "General - ER Pediatrico",
    hw_giustiniani = "HW - ER Giustiniani",
    hw_osa = "HW - ER Sant'Antonio",
    hw_pediatrico = "HW - ER Pediatrico",
    ili_giustiniani = "ILI - ER Giustiniani",
    ili_osa = "ILI - ER Sant'Antonio",
    ili_pediatrico = "ILI - ER Pediatrico"
)

driver_kind_labels <- c(
    none = "No driver",
    forced = "Forced",
    pollution = "Pollution",
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

plotted_2025 <- results %>%
    filter(testing_end == "2025-12-31") %>%
    filter(grepl(
        "20260916_214551|20260916_223834|20260916_232109|20260917_141023",
        model_file
    )) %>%
    # A facet is a single dataset, so only plot one row for each model
    # configuration represented in that panel.
    distinct(dataset_name, lag, subset, driver, .keep_all = TRUE) %>%
    group_by(dataset_name) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    ungroup()

points_2025 <- bind_rows(
    plotted_2025 %>% filter(driver_kind != "none"),
    plotted_2025 %>%
        filter(driver_kind == "none") %>%
        distinct(dataset_name, aic, mape, driver_kind, .keep_all = TRUE)
)

point_counts_2025 <- points_2025 %>%
    count(dataset_name, hospital, driver_kind, lag, name = "n_points") %>%
    arrange(dataset_name, hospital, driver_kind, lag)

baseline_2025 <- results %>%
    filter(testing_end == "2025-12-31", model_file == "") %>%
    left_join(
        plotted_2025 %>%
            group_by(dataset_name) %>%
            summarise(
                point_min = min(mape),
                point_max = max(mape), .groups = "drop"
            ),
        by = "dataset_name"
    ) %>%
    filter(!is.na(point_min), !is.na(point_max)) %>%
    mutate(
        blue_max = if_else(mape >= point_max, Inf, mape),
        red_min = if_else(mape <= point_min, -Inf, mape)
    ) %>%
    filter(model_number <= 7)

baseline_labels_2025 <- setNames(
    sprintf(
        "%s\nBaseline MAPE: %.2f%%",
        dataset_labels[baseline_2025$dataset_name],
        baseline_2025$mape
    ),
    baseline_2025$dataset_name
)

print(point_counts_2025, n = Inf)
View(point_counts_2025)

ggplot() +
    geom_rect(
        data = baseline_2025 %>% filter(mape > point_min),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = blue_max),
        fill = "lightblue",
        alpha = 0.35,
        inherit.aes = FALSE
    ) +
    geom_rect(
        data = baseline_2025 %>% filter(mape < point_max),
        aes(xmin = -Inf, xmax = Inf, ymin = red_min, ymax = Inf),
        fill = "lightcoral",
        alpha = 0.25,
        inherit.aes = FALSE
    ) +
    geom_point(
        data = points_2025 %>% filter(driver_kind != "none"),
        mapping = aes(x = delta_aic, y = mape, color = lag, shape = driver_kind),
        size = 2.5
    ) +
    geom_point(
        data = points_2025 %>% filter(driver_kind == "none"),
        mapping = aes(x = delta_aic, y = mape, shape = driver_kind),
        color = "black",
        size = 2.5
    ) +
    geom_text_repel(
        data = plotted_2025 %>%
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
        nudge_x = 2,
        hjust = 0,
        direction = "y",
        min.segment.length = 1,
        show.legend = FALSE
    ) +
    facet_wrap(
        ~dataset_name,
        scales = "free",
        labeller = ggplot2::as_labeller(baseline_labels_2025)
    ) +
    scale_color_discrete(name = "Lag (weeks)", labels = lag_labels) +
    scale_shape_manual(
        name = "Driver type",
        values = c(none = 18, weather = 16),
        labels = driver_kind_labels
    ) +
    scale_y_continuous(labels = scales::label_percent(scale = 1)) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom") +
    labs(
        # title = "MAPE vs Delta AIC by Driver Kind",
        x = expression(Delta * "AIC"),
        y = "MAPE"
    )
ggsave(
    filename = "graphs/fig3bis.pdf",
    width = 12,
    height = 8.5,
    units = "in",
)


plotted_2025 %>%
    select(model_id, dataset_name, driver_kind, driver, lag, aic, mape)

plotted_2022 <- results %>%
    filter(testing_end == "2022-12-31") %>%
    filter(driver_kind != "forced") %>%
    filter(grepl(
        "20260917_071558|20260917_084103|20260917_090906|20260917_141023|20260917_155527|20260917_161251|20260917_165410|20260917_172420",
        model_file
    )) %>%
    distinct(dataset_name, lag, subset, driver, .keep_all = TRUE) %>%
    group_by(dataset_name) %>%
    mutate(delta_aic = aic - min(aic, na.rm = TRUE)) %>%
    ungroup()

best_models_2022 <- plotted_2022 %>%
    group_by(dataset_name) %>%
    slice_min(mape, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
        model_label = coalesce(unname(driver_labels[driver]), driver)
    )

points_2022 <- bind_rows(
    plotted_2022 %>% filter(driver_kind != "none"),
    plotted_2022 %>%
        filter(driver_kind == "none") %>%
        distinct(dataset_name, aic, mape, driver_kind, .keep_all = TRUE)
)

point_counts_2022 <- points_2022 %>%
    count(dataset_name, hospital, driver_kind, lag, name = "n_points") %>%
    arrange(dataset_name, hospital, driver_kind, lag)

baseline_2022 <- results %>%
    filter(testing_end == "2022-12-31", model_file == "") %>%
    left_join(
        plotted_2022 %>%
            group_by(dataset_name) %>%
            summarise(point_min = min(mape), point_max = max(mape), .groups = "drop"),
        by = "dataset_name"
    ) %>%
    filter(!is.na(point_min), !is.na(point_max)) %>%
    mutate(
        blue_max = if_else(mape >= point_max, Inf, mape),
        red_min = if_else(mape <= point_min, -Inf, mape)
    )

baseline_labels_2022 <- dataset_labels
baseline_labels_2022[as.character(baseline_2022$dataset_name)] <- sprintf(
    "%s\nBaseline MAPE: %.2f%%",
    dataset_labels[as.character(baseline_2022$dataset_name)],
    baseline_2022$mape
)

print(point_counts_2022, n = Inf)
View(point_counts_2022)
ggplot() +
    geom_rect(
        data = baseline_2022 %>% filter(mape > point_min),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = blue_max),
        fill = "lightblue",
        alpha = 0.35,
        inherit.aes = FALSE
    ) +
    geom_rect(
        data = baseline_2022 %>% filter(mape < point_max),
        aes(xmin = -Inf, xmax = Inf, ymin = red_min, ymax = Inf),
        fill = "lightcoral",
        alpha = 0.25,
        inherit.aes = FALSE
    ) +
    geom_point(
        data = points_2022 %>% filter(driver_kind != "none"),
        mapping = aes(x = delta_aic, y = mape, color = lag, shape = driver_kind),
        size = 2.5
    ) +
    geom_point(
        data = points_2022 %>% filter(driver_kind == "none"),
        mapping = aes(x = delta_aic, y = mape, shape = driver_kind),
        color = "black",
        size = 2.5
    ) +
    geom_text_repel(
        data = best_models_2022,
        mapping = aes(x = delta_aic, y = mape, label = model_label),
        color = "black",
        size = 2.7,
        box.padding = 0.35,
        point.padding = 0.2,
        min.segment.length = 0,
        direction = "both",
        max.overlaps = Inf,
        seed = 42,
        show.legend = FALSE
    ) +
    facet_wrap(
        ~dataset_name,
        scales = "free",
        labeller = ggplot2::as_labeller(baseline_labels_2022)
    ) +
    scale_color_discrete(name = "Lag (weeks)", labels = lag_labels) +
    scale_shape_discrete(name = "Driver type", labels = driver_kind_labels) +
    scale_y_continuous(labels = scales::label_percent(scale = 1)) +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom") +
    labs(
        # title = "MAPE vs Delta AIC by Driver Kind",
        x = expression(Delta * "AIC"), y = "MAPE"
    )
ggsave(
    filename = "graphs/fig3bis_2.pdf",
    width = 12,
    height = 8.5,
    units = "in",
)
