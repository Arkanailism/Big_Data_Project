library(tidyverse)
library(countrycode)
library(tidymodels)

# 1 & 2 – load + clean + ISO‑3
energy <- read_csv("./data/global_energy_consumption.csv") %>% janitor::clean_names()
pollut <- read_csv("./data/global air pollution dataset.csv") %>% janitor::clean_names()
names(pollut)
# robust column detection ---------------------------------------------------
pc_col     <- "per_capita_energy_use_k_wh"
renew_col  <- "renewable_energy_share_percent"
fossil_col <- "fossil_fuel_dependency_percent"
pm_col     <- "pm2_5_aqi_value"

# abort with message if any required column is missing
if (any(is.na(c(pc_col, renew_col, fossil_col, pm_col))))
  stop("One or more key columns could not be found – check the raw CSV headers.")

# 3 – feature engineering ---------------------------------------------------
energy23 <- energy %>%
  filter(year == 2023) %>%
  mutate(
    iso3        = countrycode(country, "country.name", "iso3c"),
    renew_ratio = .data[[renew_col]] / .data[[fossil_col]],
    log_pc_kwh  = log10(.data[[pc_col]])
  ) %>%
  select(iso3, renew_ratio, log_pc_kwh)

# 4 – aggregation -----------------------------------------------------------
poll_cty <- pollut %>%
  mutate(iso3 = countrycode(country, "country.name", "iso3c")) %>%
  group_by(iso3) %>%
  summarise(pm25_avg = mean(.data[[pm_col]], na.rm = TRUE), .groups = "drop")

# 5 – join & split ----------------------------------------------------------
analytic <- inner_join(energy23, poll_cty, by = "iso3") %>% drop_na()

set.seed(42)
split <- initial_split(analytic, prop = 0.8, strata = pm25_avg)
train <- training(split); valid <- testing(split)

# 6 & 7 – recipe ------------------------------------------------------------
rec <- recipe(pm25_avg ~ renew_ratio + log_pc_kwh, data = train) %>%
  step_impute_median(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# model specs
gam_spec <- gen_additive_mod() %>% set_mode("regression")
rf_spec  <- rand_forest(mtry = tune(), trees = 1000, min_n = tune()) %>%
  set_engine("ranger") %>% set_mode("regression")
xgb_spec <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(),
                       mtry = tune(), sample_size = tune()) %>%
  set_engine("xgboost") %>% set_mode("regression")
lm_spec <- linear_reg() %>% set_engine("lm")

wf_gam <- workflow() %>% add_recipe(rec) %>% add_model(gam_spec)
wf_rf  <- workflow() %>% add_recipe(rec) %>% add_model(rf_spec)
wf_xgb <- workflow() %>% add_recipe(rec) %>% add_model(xgb_spec)
wf_lm <- workflow() %>% add_recipe(rec) %>% add_model(lm_spec)

folds <- vfold_cv(train, v = 10, strata = pm25_avg)

rf_grid  <- grid_regular(mtry(range = c(1,2)),
                         min_n(range = c(2,10)), levels = 5)
rf_tune  <- tune_grid(wf_rf , resamples = folds, grid = rf_grid ,
                      metrics = metric_set(rmse))

# XGBoost parameter set -----------------------------------------------------
param_set <- parameters(
  trees(), tree_depth(), learn_rate(),
  finalize(mtry(), train),  # adapts to n predictors (=2)
  sample_prop()
)
install.packages("xgboost")
library("xgboost")
xgb_grid <- grid_space_filling(param_set, size = 20)

xgb_tune <- tune_grid(wf_xgb, resamples = folds, grid = xgb_grid,
                      metrics = metric_set(rmse))

# final fits ---------------------------------------------------------------
best_rf  <- select_best(rf_tune , metric = "rmse")
best_xgb <- select_best(xgb_tune, metric = "rmse")

final_rf  <- finalize_workflow(wf_rf , best_rf )  %>% fit(train)
final_xgb <- finalize_workflow(wf_xgb, best_xgb) %>% fit(train)
final_lm  <- wf_lm %>% fit(train)
final_xgb <- finalize_workflow(wf_xgb, select_best(xgb_tune, "rmse")) %>% fit(train)
final_lm  <- wf_lm %>% fit(train)
pred <- valid %>%
  bind_cols(predict(final_lm , valid) %>% rename(pred_lm  = .pred),
            predict(final_rf , valid) %>% rename(pred_rf  = .pred),
            predict(final_xgb, valid) %>% rename(pred_xgb = .pred))

library(yardstick)
results <- bind_rows(
  metrics(pred, truth = pm25_avg, estimate = pred_lm ) %>% mutate(model = "Linear"),
  metrics(pred, truth = pm25_avg, estimate = pred_rf ) %>% mutate(model = "Random Forest"),
  metrics(pred, truth = pm25_avg, estimate = pred_xgb) %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
print(results)

library(yardstick)

# --- fit default (untuned) models -----------------------------------------
rf_default_spec <- rand_forest(mode = "regression") %>% set_engine("ranger")
xgb_default_spec <- boost_tree(mode = "regression") %>% set_engine("xgboost")

rf_default <- workflow() %>% add_recipe(rec) %>% add_model(rf_default_spec) %>% fit(train)
xgb_default <- workflow() %>% add_recipe(rec) %>% add_model(xgb_default_spec) %>% fit(train)

# --- predictions on validation set ----------------------------------------
pred_tbl <- valid %>%
  bind_cols(
    predict(rf_default , valid) %>% rename(rf_pre  = .pred),
    predict(final_rf   , valid) %>% rename(rf_post = .pred),
    predict(xgb_default, valid) %>% rename(xgb_pre = .pred),
    predict(final_xgb  , valid) %>% rename(xgb_post= .pred)
  )

# helper to compute metrics -------------------------------------------------
get_metrics <- function(truth, estimate) {
  metrics(pred_tbl, truth = {{ truth }}, estimate = {{ estimate }}) %>%
    select(.metric, .estimate)
}

metric_df <- bind_rows(
  get_metrics(pm25_avg, rf_pre ) %>% mutate(model = "Random Forest", stage = "Pre‑tune"),
  get_metrics(pm25_avg, rf_post) %>% mutate(model = "Random Forest", stage = "Post‑tune"),
  get_metrics(pm25_avg, xgb_pre) %>% mutate(model = "XGBoost"     , stage = "Pre‑tune"),
  get_metrics(pm25_avg, xgb_post)%>% mutate(model = "XGBoost"     , stage = "Post‑tune")
) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(model, stage)

print(metric_df)
