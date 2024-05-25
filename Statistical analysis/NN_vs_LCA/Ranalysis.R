library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(broom)

data <- read.csv("D:/PythonProjects/Thesis/simulations/compiled_data.csv", header = TRUE)

data_long <- pivot_longer(
  data,
  cols = starts_with(c("LCA", "NN")),
  names_to = c("type", "data_type", "counter"),
  names_pattern = "([^_]*)_([^_]*)_(\\d+)",
  values_to = "values"
)
nested_data <- data_long %>%
  mutate(counter = as.integer(counter)) %>%
  arrange(type, data_type, counter) %>%
  group_by(type, data_type) %>%
  nest()

# This creates a nested tibble with one column being a list of tibbles, each containing the 'values'
nested_list <- map(nested_data$data, ~ {
  split(.$values, .$counter)  # Split the nested tibble into lists by 'counter'
})

names(nested_list) <- paste(nested_data$type, nested_data$data_type, sep = "$")
# Create a more complex nested list for direct access
final_list <- list()
for (type in unique(data_long$type)) {
  final_list[[type]] <- list()
  type_data <- nested_list[grep(type, names(nested_list))]
  
  for (data_type in unique(data_long$data_type)) {
    final_list[[type]][[data_type]] <- type_data[grep(data_type, names(type_data))]
  }
}

#Plotting

calculate_summary_stats <- function(data_df) {
  summary_stats <- data_df %>%
    rowwise() %>%
    mutate(
      average = mean(c_across(where(is.numeric)), na.rm = TRUE),
      std_dev1 = sd(c_across(where(is.numeric))[c_across(where(is.numeric)) > average], na.rm = TRUE),
      std_dev2 = sd(c_across(where(is.numeric))[c_across(where(is.numeric)) < average], na.rm = TRUE),
      lower_ci = average - qt(0.975, df = sum(c_across(where(is.numeric)) < average, na.rm = TRUE) - 1) * (std_dev2 / sqrt(sum(c_across(where(is.numeric)) < average, na.rm = TRUE))),
      upper_ci = average + qt(0.975, df = sum(c_across(where(is.numeric)) > average, na.rm = TRUE) - 1) * (std_dev1 / sqrt(sum(c_across(where(is.numeric)) > average, na.rm = TRUE))),
      max = max(c_across(where(is.numeric)), na.rm = TRUE),
      min = min(c_across(where(is.numeric)), na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(simulations = row_number())  # Add a row number to each entry
  
  return(summary_stats)
}

plot_data <- final_list$LCA$Training

# Convert to a data frame
data_df <- calculate_summary_stats(as.data.frame(plot_data))


ggplot(data_df, aes(x = simulations, y = average)) +
  geom_line() +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "blue") +
  labs(title = "Fitness of LCA on Training",
       x = "Simulations",
       y = "Fitness") +
  theme_minimal() +
  ylim(0, 1)



NN_training_df <- calculate_summary_stats(as.data.frame(final_list$NN$Training))
LCA_training_df <- calculate_summary_stats(as.data.frame(final_list$LCA$Training))
NN_control_df <- calculate_summary_stats(as.data.frame(final_list$NN$Control))
LCA_control_df <- calculate_summary_stats(as.data.frame(final_list$LCA$Control))

comparison_df <- data.frame(simulations = vector("numeric", length = 1000))
comparison_df$simulations <- NN_training_df$simulations
comparison_df$NN_training_average <- NN_training_df$average
comparison_df$LCA_training_average <- LCA_training_df$average
comparison_df$NN_control_average <- NN_control_df$average
comparison_df$LCA_control_average <- LCA_control_df$average

comparison_df$NN_training_max <- NN_training_df$max
comparison_df$LCA_training_max <- LCA_training_df$max
comparison_df$NN_control_max <- NN_control_df$max
comparison_df$LCA_control_max <- LCA_control_df$max

comparison_df$NN_training_min <- NN_training_df$min
comparison_df$LCA_training_min <- LCA_training_df$min
comparison_df$NN_control_min <- NN_control_df$min
comparison_df$LCA_control_min <- LCA_control_df$min

library(ggplot2)

# Corrected ggplot code
ggplot(comparison_df, aes(x = simulations)) +
  geom_smooth(aes(y = NN_control_min, colour = "NN"), se = FALSE) +
  geom_smooth(aes(y = NN_control_average, colour = "NN"), se = FALSE) +
  geom_smooth(aes(y = NN_control_max, colour = "NN"), se = FALSE) +
  geom_smooth(aes(y = LCA_control_min, colour = "LCA"), se = FALSE) +
  geom_smooth(aes(y = LCA_control_average, colour = "LCA"), se = FALSE) +
  geom_smooth(aes(y = LCA_control_max, colour = "LCA"), se = FALSE) +
  scale_color_manual(values = c(
    "NN" = "#FF0000",   
    "LCA" = "#0000FF"
  )) +
  labs(title = "Comparison of min, max and average of NN and LCA",
       x = "Simulations",
       y = "Fitness") 

