# ----------------------------- Libraries --------------------------------------

library(data.table)
library(tidyverse)
library(lubridate)
library(skimr)
library(inspectdf)
library(timetk)
library(highcharter)
library(modeltime.h2o)
library(tidymodels)



# ----------------------------- Data preparation -------------------------------

# Read Data
raw <- fread("daily-minimum-temperatures-in-me.csv")


# Check names & datatypes of columns
raw %>% glimpse()
df <- copy(raw)


# Change names & datatypes of columns
colnames(df) <- c("Date", "Temp")
df$Date <- as.Date(parse_date_time(df$Date, "mdy"))
df$Temp <- df$Temp %>% as.numeric()
df %>% skim()


# Check NA values
inspect_na(df)
df[!complete.cases(df)]


# Drop NA values
df <- df[!is.na(df$Temp)]



# ----------------------------- Exploratory Analysis ---------------------------

# Outlier Detection
df %>%
  plot_anomaly_diagnostics(
    .date = Date,
    .value = Temp,
    .facet_ncol = 2,
    .interactive = T,
    .title = "Anomaly Diagnostics",
    .anom_color = "#FB3029",
    .max_anomalies = 0.07,
    .alpha = 0.05
  )


# Seasonality plots
df %>%
  plot_seasonal_diagnostics(
    Date, Temp,
    .interactive = T
  )


# Autocorrelation and Partial Autocorrelation
df %>%
  plot_acf_diagnostics(
    Date, Temp,
    .lags = "1 year", .interactive = T
  )



# ----------------------------- Feature Engineering ----------------------------

# Extract all possible features
all_time_arg <- df %>% tk_augment_timeseries_signature(.date_var = Date)
all_time_arg %>% skim()

df <- all_time_arg %>%
  select(-hour, -hour12, -minute, -second, -am.pm) %>%
  mutate_if(is.ordered, as.character) %>%
  mutate_if(is.character, as_factor)

df %>% view()



# ----------------------------- H2O --------------------------------------------

# H2O initialization
h2o.init()


# Transform dataframe
train_h2o <- df %>%
  filter(year < 1990) %>%
  as.h2o()
test_h2o <- df %>%
  filter(year >= 1990) %>%
  as.h2o()

y <- "Temp"
x <- df %>%
  select(-Temp) %>%
  names()


# -------------- Train models ---------------
model_h2o <- h2o.automl(
  x = x, y = y,
  training_frame = train_h2o,
  validation_frame = test_h2o,
  leaderboard_frame = test_h2o,
  stopping_metric = "RMSE",
  seed = 123, nfolds = 10,
  max_runtime_secs = 120
)


# Show the leader board
model_h2o@leaderboard %>% as.data.frame()


# Select best performed model
h2o_leader <- model_h2o@leader


# -------------- Evaluate ---------------
h2o_leader %>%
  h2o.rmse(train = T, valid = T, xval = T)


# Predict
pred_h2o <- h2o_leader %>% h2o.predict(test_h2o)


# Error table
error_tbl <- df %>%
  filter(lubridate::year(Date) >= 1990) %>%
  add_column(pred = pred_h2o %>% as_tibble() %>% pull(predict)) %>%
  rename(actual = Temp) %>%
  select(Date, actual, pred)


# Plot prediction
highchart() %>%
  hc_xAxis(categories = error_tbl$Date) %>%
  hc_add_series(
    data = error_tbl$actual, type = "line",
    color = "red", name = "Actual"
  ) %>%
  hc_add_series(
    data = error_tbl$pred, type = "line",
    color = "green", name = "Predicted"
  ) %>%
  hc_title(text = "Predict")



# -------------------------------- Tidy models --------------------------------

train <- df %>% filter(year < 1990)
test <- df %>% filter(year >= 1990)

# Auto ARIMA
model_fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(Temp ~ Date, train)


calibration <- modeltime_table(model_fit_arima) %>%
  modeltime_calibrate(test)


# Predict
calibration %>%
  modeltime_forecast(actual_data = df) %>%
  plot_modeltime_forecast(
    .interactive = T,
    .plotly_slider = T
  )


# Accuracy
calibration %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = F)



# ----------------------------- New data ---------------------------------------

# -------------- New data (next 1 years) ---------------
new_data <- seq(as.Date("1991/01/01"), as.Date("1991/12/31"), "days") %>%
  as_tibble() %>%
  add_column(Temp = 0) %>%
  rename(Date = value) %>%
  tk_augment_timeseries_signature(.date_var = Date) %>%
  select(-hour, -hour12, -minute, -second, -am.pm) %>%
  mutate_if(is.ordered, as.character) %>%
  mutate_if(is.character, as_factor)


# -------------- Forecast ---------------

# H2O
new_h2o <- new_data %>% as.h2o()

new_predictions <- h2o_leader %>%
  h2o.predict(new_h2o) %>%
  as_tibble() %>%
  add_column(Date = new_data$Date) %>%
  select(Date, predict) %>%
  rename(Temp = predict)

bind_rows(df, new_predictions) %>%
  mutate(colors = c(rep("Actual", 3647), rep("Predicted", 365))) %>%
  hchart("line", hcaes(Date, Temp, group = colors)) %>%
  hc_title(text = "Forecast") %>%
  hc_colors(colors = c("red", "green"))


# Auto ARIMA
calibration %>%
  modeltime_refit(df) %>%
  modeltime_forecast(
    h = "1 year",
    actual_data = df
  ) %>%
  select(-contains("conf")) %>%
  plot_modeltime_forecast(
    .interactive = T,
    .plotly_slider = T,
    .legend_show = F
  )