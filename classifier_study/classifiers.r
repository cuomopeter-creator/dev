
#declaring packages needed for the project, installing if not present, and loading them
needed_pkgs <- c(
  "dplyr", "ggplot2", "caret", "Rtsne",
  "rpart", "rpart.plot",
  "RWeka"
)

#installing packages if not already installed
to_install <- needed_pkgs[!sapply(needed_pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install)

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(caret)
  library(Rtsne)
  library(rpart)
  library(rpart.plot)
  library(RWeka)
})

set.seed(42)


#Reading the dataset
df_raw <- read.csv("credit-g.csv", stringsAsFactors = FALSE)

cat("Rows:", nrow(df_raw), "Cols:", ncol(df_raw), "\n")
cat("Target column expected: class\n")
stopifnot("class" %in% names(df_raw))



# ---------------------------
# Task 1: Preprocessing
# ---------------------------


df <- df_raw

#Clean up quoted categorical strings 
clean_quotes <- function(x) {
    #triming whitespace first to avoid issues with quotes being separated by spaces
  x <- trimws(x)
  # remove leading/trailing single quotes
  x <- gsub("^'+|'+$", "", x)
    #trim again in case there were spaces around the quotes
  x <- trimws(x)
  x
}

# Apply to character columns by first finding them, then applying the function we just created.
char_cols <- names(df)[sapply(df, is.character)]
for (cn in char_cols) {
  df[[cn]] <- clean_quotes(df[[cn]])
}


# (B) Convert target to factor (caret expects factor for classification)
df$class <- factor(df$class)

#checking the frequency of each class in the target variable to understand the class distribution
cat("Class distribution (raw):\n")
print(table(df$class))


# (C) Missing value handling (robust even if none exist)
# Numeric: median impute; Factor: mode impute
mode_impute <- function(v) {
    #makes copy of the vector without NA values to calculate the mode
  v2 <- v[!is.na(v)]
  if (length(v2) == 0) return(v)
  #mode i.e. most frequent value
  m <- names(sort(table(v2), decreasing = TRUE))[1]
  v[is.na(v)] <- m
  v
}

for (cn in names(df)) {
  if (cn == "class") next
  # Check if numeric/integer => median impute; else mode impute
  if (is.numeric(df[[cn]]) || is.integer(df[[cn]])) {
    if (any(is.na(df[[cn]]))) {
      med <- median(df[[cn]], na.rm = TRUE)
      df[[cn]][is.na(df[[cn]])] <- med
    }
    # If no NAs, do nothing (leave as is)
  } else {
    if (any(is.na(df[[cn]]))) {
        #mode imputation for categorical predictors
      df[[cn]] <- mode_impute(df[[cn]])
    }
  }
}

# (D) Convert remaining non-numeric predictors to factors
for (cn in names(df)) {
  if (cn == "class") next
  if (!is.numeric(df[[cn]]) && !is.integer(df[[cn]])) {
    df[[cn]] <- factor(df[[cn]])
  }
}

# (E) Remove near-zero variance predictors (safe housekeeping)
nzv <- nearZeroVar(df %>% select(-class))
# If any near-zero variance predictors are found, print their names and remove them from the dataset.
if (length(nzv) > 0) {
  cat("Removing near-zero variance columns:\n")
  print(names(df %>% select(-class))[nzv])
  df <- df[, -nzv, drop = FALSE]
} else {
  cat("No near-zero variance predictors found.\n")
}

cat("\nPreprocessing complete.\n")
cat("Final columns:", ncol(df), "\n")

# ---------------------------
# Task 1 (continued): t-SNE visualization
# ---------------------------

cat("\nTask 1 - t-SNE visualization.\n")
# t-SNE needs numeric matrix => one-hot encode factors then scale
x <- df %>% select(-class)
y <- df$class

dv <- dummyVars(~ ., data = x, fullRank = TRUE)
x_mat <- predict(dv, newdata = x)

# Scale features for t-SNE stability
x_scaled <- scale(x_mat)

# Rtsne settings (n=1000 so perplexity 30 is fine)
tsne_out <- Rtsne(
  x_scaled,
  dims = 2,
  #tested perplexity values: 5, 10, 20, 30, 50; 50 gave best visual separation without too much noise
  perplexity = 50,
  verbose = TRUE,
  max_iter = 1000,
  check_duplicates = FALSE
)
# Create a data frame for ggplotting
tsne_df <- data.frame(
  TSNE1 = tsne_out$Y[, 1],
  TSNE2 = tsne_out$Y[, 2],
  class = y
)
# Plot t-SNE with ggplot2
p <- ggplot(tsne_df, aes(TSNE1, TSNE2, color = class)) +
  geom_point(alpha = 0.75, size = 2) +
  labs(
    title = "t-SNE projection (German credit)",
    subtitle = "Points colored by class",
    x = "t-SNE 1", y = "t-SNE 2"
  ) +
  theme_minimal()

print(p)

# Save plot for report
ggsave("task1_tsne_plot.png", plot = p, width = 8, height = 5, dpi = 300)
cat("Saved t-SNE plot to: task1_tsne_plot.png\n")



# ---------------------------
# Task 2: Train classifiers + extract rule bases
# ---------------------------

# Consistent repeated CV folds (10-fold, 3 repeats => 30 resamples)
folds <- createMultiFolds(y = df$class, k = 10, times = 3)

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  index = folds,
  classProbs = FALSE,
  savePredictions = "final",
  verboseIter = FALSE
)

#declaring the metric to optimize during training (accuracy in this case)
metric <- "Accuracy"

# ---- Decision Tree (rpart)
cat("\nTraining Decision Tree (rpart)...\n")
model_dt <- train(
  class ~ .,
  data = df,
  method = "rpart",
  trControl = ctrl,
  metric = metric,
  #tuneLength = 10 means caret will try 10 different values of the complexity parameter (cp) to find the best one based on CV performance
  tuneLength = 10
)

cat("Decision Tree done.\n")
print(model_dt)
#roughly 70% accuracy, which is decent for this dataset
# The rpart classifier achieved its highest cross-validated performance at a complexity 
# parameter of approximately 0.01, indicating an optimal balance between model complexity and generalization. 
# Both accuracy and Cohen’s Kappa decline for larger cp values, suggesting that excessive pruning reduces predictive power.

# ---- PART (RWeka)
cat("\nTraining PART (RWeka)...\n")
model_part <- train(
  class ~ .,
  data = df,
  method = "PART",
  trControl = ctrl,
  metric = metric,
  tuneLength = 1
)

cat("PART done.\n")
print(model_part)


# ---- Ripper
cat("\nTraining Ripper ...\n")
model_rip <- train(
  class ~ .,
  data = df,
  method = "JRip",
  trControl = ctrl,
  metric = metric,
  tuneLength = 1
)

cat("JRip done.\n")
print(model_rip)

# The JRip classifier achieved an average accuracy of 70.5% under 10-fold cross-validation with 
# 3 repeats; however, the low Kappa value (0.23) indicates limited predictive power beyond chance
# , reflecting the inherent difficulty and class overlap within the German credit dataset.

# Quick sanity check: confusion matrices on full data predictions

pred_dt   <- predict(model_dt, newdata = df)
pred_part <- predict(model_part, newdata = df)
pred_rip  <- predict(model_rip, newdata = df)


print(confusionMatrix(pred_dt, df$class))


print(confusionMatrix(pred_part, df$class))


print(confusionMatrix(pred_rip, df$class))

# ---------------------------
# Task 2 Part 2: Extract rule bases
# ---------------------------


# Decision tree rule base
# 1) Plot tree to file
png("task2_decision_tree_plot.png", width = 1200, height = 800)
rpart.plot(model_dt$finalModel, main = "Decision Tree (rpart)")
dev.off()
print("Saved decision tree plot to: task2_decision_tree_plot.png\n")

# 2) Extract human-readable rules (rpart)
dt_rules <- capture.output(rpart.rules(model_dt$finalModel, roundint = FALSE))
writeLines(dt_rules, "task2_rules_decision_tree.txt")
print("Saved Decision Tree rules to: task2_rules_decision_tree.txt\n")

# PART rules
part_rules <- capture.output(model_part$finalModel)
writeLines(part_rules, "task2_rules_PART.txt")
print("Saved PART rules to: task2_rules_PART.txt\n")

# JRip rules
rip_rules <- capture.output(model_rip$finalModel)
writeLines(rip_rules, "task2_rules_Ripper_JRip.txt")
print("Saved Ripper/JRip rules to: task2_rules_Ripper_JRip.txt\n")



# ---------------------------
# Task 3: Collect accuracies across 30 resamples + ANOVA + Tukey
# ---------------------------

# Each $resample is length 30 (10 folds * 3 repeats)
acc_dt   <- model_dt$resample$Accuracy
acc_part <- model_part$resample$Accuracy
acc_rip  <- model_rip$resample$Accuracy

print("Length of accuracy vectors:")
#verifying that we have 30 accuracy values for each model, which corresponds to the 30 resamples from our repeated CV setup
cat("DT:", length(acc_dt), " PART:", length(acc_part), " JRip:", length(acc_rip))

cat("\nSummary accuracies:\n")
#calculating and printing the mean and standard deviation of the accuracy for each model across the 30 resamples to understand their average performance and variability
cat("DT mean:", mean(acc_dt), "sd:", sd(acc_dt), "\n")
cat("PART mean:", mean(acc_part), "sd:", sd(acc_part), "\n")
cat("JRip mean:", mean(acc_rip), "sd:", sd(acc_rip), "\n")

# Build long format for ANOVA
acc_df <- data.frame(
  accuracy = c(acc_dt, acc_part, acc_rip),
  model = factor(rep(c("DecisionTree", "PART", "Ripper_JRip"),
                     times = c(length(acc_dt), length(acc_part), length(acc_rip))))
)

# One-way ANOVA (F-test)
anova_fit <- aov(accuracy ~ model, data = acc_df)
anova_tbl <- summary(anova_fit)

cat("\nANOVA table:\n")
print(anova_tbl)

# Saving ANOVA table to file for report
anova_out <- capture.output(anova_tbl)
writeLines(anova_out, "task3_anova_table.txt")


# If significant at 0.05 => Tukey HSD
pval <- anova_tbl[[1]][["Pr(>F)"]][1]
cat("\nANOVA p-value:", pval, "\n")
# The ANOVA p-value of approximately 0.03 indicates a statistically significant difference in mean accuracy among the three classifiers at the 95% confidence level, suggesting that at least one model performs differently from the others on the German credit dataset.
if (!is.na(pval) && pval < 0.05) {
  cat("\nSignificant difference detected (p < 0.05). Running Tukey HSD...\n")
  tuk <- TukeyHSD(anova_fit)
  print(tuk)

  tuk_out <- capture.output(tuk)
  writeLines(tuk_out, "task3_tukeyHSD.txt")
  cat("Saved Tukey HSD output to: task3_tukeyHSD.txt\n")

  # Identify highest mean accuracy model
  means <- tapply(acc_df$accuracy, acc_df$model, mean)
  best_model <- names(which.max(means))
  cat("\nHighest mean accuracy model:", best_model, "\n")
  cat("Means:\n")
  print(means)

} else {
  cat("\nNo significant difference detected at 95% confidence (p >= 0.05).\n")
  cat("Tukey test not required per project instructions.\n")
}
