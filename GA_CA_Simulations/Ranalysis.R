data <- read.csv("summary_data.csv")

extract_reference_levels <- function(column) {
  if(is.factor(column)) {
    return(levels(column)[1])  # Return the first level (reference level)
  }
}

# Ensure correct types for continuous variables
data$crossover_rate <- as.numeric(data$crossover_rate)
data$population_size <- as.numeric(data$population_size)
data$input_percentage <- as.numeric(data$input_percentage)
data$CA_size <- as.numeric(data$CA_size)
data$steps <- as.numeric(data$steps)
data$mutation_rate <- as.numeric(data$mutation_rate)
data$mutation_std_dev <- as.numeric(data$mutation_std_dev)
data$selection_size <- as.numeric(data$selection_size)
data$growth_rate_mean_fitness_training <- as.numeric(data$growth_rate_mean_fitness_training)
data$growth_rate_mean_fitness_control <- as.numeric(data$growth_rate_mean_fitness_control)
data$growth_rate_best_fitness_control <- as.numeric(data$growth_rate_best_fitness_control)
data$growth_rate_best_fitness_training <- as.numeric(data$growth_rate_best_fitness_training)


# Remove variables with any missing values
data <- data[, colSums(is.na(data)) == 0]

# Start capturing output to a file
sink("anova_results.txt")

# Example for growth_rate_mean_fitness_training
cat("ANOVA Results for Growth Rate Mean Fitness Training:\n")

# Get all column names except the last four
predictors <- names(data)[1:(ncol(data) - 4)]

# Create the formula string dynamically
# 'growth_rate_mean_fitness_training' is your response variable, it's fixed in this example
formula_string <- paste("growth_rate_mean_fitness_training ~", paste(predictors, collapse=" + "))

# Convert the string to a formula
formula <- as.formula(formula_string)

# Perform the ANOVA
aov_result_training <- aov(formula, data = data)
summary_aov_training <- summary(aov_result_training)
print(summary_aov_training)
cat("\n\n")

# Get indices of significant predictors
significant_indices <- which(summary_aov_training[[1]][,"Pr(>F)"] < 0.1)

# Use indices to get names
significant_predictors <- rownames(summary_aov_training[[1]])[significant_indices]

# Build formula string for significant predictors only
significant_formula <- paste("growth_rate_mean_fitness_training ~", paste(significant_predictors, collapse=" + "))
print(significant_formula)

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

# Repeat similar steps for growth_rate_mean_fitness_control if necessary
# Example for growth_rate_mean_fitness_training
cat("ANOVA Results for Growth Rate Mean Fitness Control:\n")
formula_string <- paste("growth_rate_mean_fitness_control ~", paste(predictors, collapse=" + "))

# Convert the string to a formula
formula <- as.formula(formula_string)

# Perform the ANOVA
aov_result_control <- aov(formula, data = data)
summary_aov_control <- summary(aov_result_control)
print(summary_aov_control)
cat("\n\n")

# Identify significant predictors with p-value < 0.1
# Get indices of significant predictors
significant_indices <- which(summary_aov_control[[1]][,"Pr(>F)"] < 0.1)

# Use indices to get names
significant_predictors <- rownames(summary_aov_training[[1]])[significant_indices]


# Build formula string for significant predictors only
significant_formula <- paste("growth_rate_mean_fitness_control ~", paste(significant_predictors, collapse=" + "))

# Fit a linear model using only significant predictors
lm_result_significant <- lm(as.formula(significant_formula), data=data)

# Display the model summary
cat("Linear Model Results for Growth Rate Mean Fitness Control using Significant Predictors Only:\n")
print(summary(lm_result_significant))
cat("\n\nReference Levels for Categorical Variables:\n")
# This function will be applied to each column in the dataframe
# Apply the function to each column and filter out NULLs
reference_levels <- sapply(data, extract_reference_levels)
reference_levels <- reference_levels[!sapply(reference_levels, is.null)]

# Print the names of variables alongside their reference levels
print(reference_levels)
cat("\n\n")

# Stop capturing output to a file
sink()
