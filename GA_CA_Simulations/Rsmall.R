data <- read.csv("summary_data.csv")

data$growth_rate_mean_fitness_training <- as.numeric(data$growth_rate_mean_fitness_training)
data$growth_rate_mean_fitness_control <- as.numeric(data$growth_rate_mean_fitness_control)
data$growth_rate_best_fitness_control <- as.numeric(data$growth_rate_best_fitness_control)
data$growth_rate_best_fitness_training <- as.numeric(data$growth_rate_best_fitness_training)

# Start capturing output to a file
sink("anova_results_other_classification.txt")

# Example for growth_rate_mean_fitness_training
cat("ANOVA Results for Growth Rate Mean Fitness Training:\n")

# Perform the ANOVA
aov_result_training <- aov(growth_rate_mean_fitness_training ~ sort_rules, data = data)
summary_aov_training <- summary(aov_result_training)
print(summary_aov_training)
cat("\n\n")

# Fit a linear model using only significant predictors
lm_result_significant <- lm(growth_rate_mean_fitness_training ~ sort_rules, data=data)

# Display the model summary
cat("Linear Model Results for Growth Rate Mean Fitness Training using Significant Predictors Only:\n")
print(summary(lm_result_significant))
cat("\n\nReference Levels for Categorical Variables:\n")
reference_levels <- sapply(data, extract_reference_levels)
reference_levels <- reference_levels[!sapply(reference_levels, is.null)]
# Print the names of variables alongside their reference levels
print(reference_levels)
cat("\n\n")

# Example for growth_rate_mean_fitness_training
cat("ANOVA Results for Growth Rate Mean Fitness Control:\n")

# Perform the ANOVA
aov_result_control <- aov(growth_rate_mean_fitness_control ~ sort_rules, data = data)
summary_aov_control <- summary(aov_result_control)
print(summary_aov_control)
cat("\n\n")

# Fit a linear model using only significant predictors
lm_result_significant <- lm(as.formula(significant_formula), data=data)

# Display the model summary
cat("Linear Model Results for Growth Rate Mean Fitness Training using Significant Predictors Only:\n")
print(summary(lm_result_significant))
cat("\n\nReference Levels for Categorical Variables:\n")
reference_levels <- sapply(data, extract_reference_levels)
reference_levels <- reference_levels[!sapply(reference_levels, is.null)]
# Print the names of variables alongside their reference levels
print(reference_levels)
cat("\n\n")

# Fit a linear model using only significant predictors
lm_result_significant <- lm(growth_rate_mean_fitness_training ~ sort_rules, data=data)

# Display the model summary
cat("Linear Model Results for Growth Rate Mean Fitness Training using Significant Predictors Only:\n")
print(summary(lm_result_significant))
cat("\n\nReference Levels for Categorical Variables:\n")
reference_levels <- sapply(data, extract_reference_levels)
reference_levels <- reference_levels[!sapply(reference_levels, is.null)]
# Print the names of variables alongside their reference levels
print(reference_levels)
cat("\n\n")

sink()