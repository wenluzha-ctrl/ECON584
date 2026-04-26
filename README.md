# ECON584
R language based code
packages <- c("MASS","pscl","car","lmtest","sandwich","AER","ggplot2","visreg","broom","modelsummary","stargazer")
install_if_missing <- function(pk) if(!require(pk, character.only = TRUE)) install.packages(pk, repos="https://cloud.r-project.org")
invisible(sapply(packages, install_if_missing))
library(MASS); library(pscl); library(car); library(lmtest); library(sandwich)
library(AER); library(ggplot2); library(visreg); library(broom); library(modelsummary); library(stargazer)

# --- 1. Create data.frame from sample (10 rows from your image) ---
df <- data.frame(
  Zipcode    = c(90024,90049,90067,90210,90036,90046,90025,90034,90019,90016),
  Stores     = c(3,2,2,1,2,1,1,0,0,0),
  Income     = c(95000,130000,110000,150000,85000,90000,80000,70000,65000,60000),
  HousePrice = c(1200000,1800000,1500000,2500000,1000000,1100000,950000,800000,750000,700000),
  Age        = c(35,45,40,50,38,37,34,33,36,32),
  Density    = c(12000,8000,9000,6000,11000,9500,10000,10500,9800,10200),
  Rent       = c(2800,3500,3200,4000,2700,2900,2600,2400,2300,2200),
  WalkScore  = c(85,70,75,60,88,82,90,87,80,78),
  Parking    = c(1,1,1,1,0,0,0,0,1,1),
  Competitor = c(1,1,1,0,1,1,1,0,0,0)
)

# Quick look
print(df)
summary(df)

# --- 2. Fit Poisson model and test overdispersion ---
pois_mod <- glm(Stores ~ Income + Density + Age + WalkScore + Parking + Competitor,
                family = poisson(link = "log"), data = df)
cat("Poisson model summary:\n"); print(summary(pois_mod))
cat("\nDispersion test (H0: equidispersion):\n"); print(dispersiontest(pois_mod))

# --- 3. Fit Negative Binomial model ---
nb_mod <- glm.nb(Stores ~ Income + Density + Age + WalkScore + Parking + Competitor,
                 data = df)
cat("\nNegative Binomial model summary:\n"); print(summary(nb_mod))

# --- 4. IRR table (estimates exponentiated) ---
est <- coef(nb_mod)
se <- sqrt(diag(vcov(nb_mod)))
irr <- exp(est)
irr_ci_low  <- exp(est - 1.96*se)
irr_ci_high <- exp(est + 1.96*se)
irr_table <- data.frame(Estimate = est, SE = se, IRR = irr, CI_lower = irr_ci_low, CI_upper = irr_ci_high)
print(round(irr_table,4))

# --- 5. Robust SEs ---
robust_vcov <- vcovHC(nb_mod, type = "HC1")
cat("\nCoefficients with robust SEs:\n"); print(coeftest(nb_mod, vcov. = robust_vcov))

# --- 6. Model comparison & zero-inflation check ---
cat("\nLR test (Poisson vs NB):\n"); print(lrtest(pois_mod, nb_mod))
zinb_mod <- zeroinfl(Stores ~ Income + Density + Age + WalkScore + Parking + Competitor | 1,
                     dist = "negbin", data = df)
cat("\nVuong test (ZINB vs NB):\n"); print(vuong(zinb_mod, nb_mod))

# --- 7. Diagnostics and plots ---
# Diagnostic plots (base)
par(mfrow = c(2,2)); plot(nb_mod); par(mfrow = c(1,1))

# Pearson residuals histogram
resid_df <- data.frame(fitted = fitted(nb_mod), pearson = residuals(nb_mod, type = "pearson"))
ggplot(resid_df, aes(x = pearson)) +
  geom_histogram(bins = 10, fill = "gray", color = "black") +
  labs(title = "Pearson residuals (NB)", x = "Pearson residual", y = "Count")

# Partial effect plots (gg)
p1 <- visreg(nb_mod, "Income", scale = "response", gg = TRUE) + ggtitle("Predicted Stores vs Income")
p2 <- visreg(nb_mod, "Density", scale = "response", gg = TRUE) + ggtitle("Predicted Stores vs Density")
print(p1); print(p2)

# Predicted counts across Income (others at mean)
newdata <- data.frame(
  Income = seq(min(df$Income), max(df$Income), length.out = 50),
  Density = mean(df$Density),
  Age = mean(df$Age),
  WalkScore = mean(df$WalkScore),
  Parking = round(mean(df$Parking)),    # keep as typical category
  Competitor = round(mean(df$Competitor))
)
newdata$predicted <- predict(nb_mod, newdata = newdata, type = "response")
ggplot(newdata, aes(x = Income, y = predicted)) + geom_line() + labs(y = "Predicted store count")

# --- 8. Multicollinearity ---
vif_vals <- vif(lm(Stores ~ Income + Density + Age + WalkScore + Parking + Competitor, data = df))
cat("\nVIFs:\n"); print(round(vif_vals,3))

# --- 9. Stepwise model selection (optional) ---
step_nb <- stepAIC(nb_mod, direction = "both", trace = FALSE)
cat("\nStepwise-selected model summary:\n"); print(summary(step_nb))

# --- 10. Formatted regression tables ---
# stargazer (console)
stargazer(nb_mod, type = "text", title = "Negative Binomial regression (sample)", single.row = TRUE, digits = 3)

# broom tidy (exponentiate coefficients to IRR)
tidy_nb <- broom::tidy(nb_mod, conf.int = TRUE)
tidy_nb <- transform(tidy_nb, IRR = exp(estimate),
                     IRR_CI_low = exp(conf.low), IRR_CI_high = exp(conf.high))
print(tidy_nb)

# modelsummary (console)
modelsummary(list(NegBin = nb_mod, StepAIC = step_nb), output = "console")

# Save tables (optional)
write.csv(irr_table, "nb_irr_table_sample.csv", row.names = TRUE)
write.csv(tidy_nb, "nb_tidy_results_sample.csv", row.names = FALSE)

cat("\nScript complete. Check console for outputs and plots in RStudio's Plots pane.\n")
