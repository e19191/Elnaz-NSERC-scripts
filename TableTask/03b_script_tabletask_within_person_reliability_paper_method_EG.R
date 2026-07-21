#!/usr/bin/env Rscript

# 03b_script_tabletask_within_person_reliability_paper_method_EG.R
#
# TableTask paper-based within-participant reliability analysis.
#
# This script adapts the time-sensitive and random split-half workflow in:
# Dudarev et al. (2023), Sensors, 23, 5863
# and the authors' released R code:
#   wearables_reliability.R
#   wearables_reliability_for_Biostrap_data.R
#
# IMPORTANT
# ---------
# This is a separate study-level reliability analysis. It does NOT replace
# Step 3a's trial-level odd-even RR consistency QC.
#
# Paper-consistent core workflow:
#   1. Order observations by acquisition time.
#   2. Within participant x repeated situation, split observations into
#      odd- and even-ordered samples.
#   3. Average each half.
#   4. Fit: odd_half_mean ~ even_half_mean + (1 | participant)
#   5. Use the fixed-effect slope beta as the within-participant reliability estimate.
#   6. Repeat with random split-halves 1000 times as a sensitivity analysis.
#
# TableTask adaptation:
#   - participant = the actual participant ID carried in each paired Step 3 file
#   - repeated situation = one participant's one TableTask trial
#   - physiological measure = Step 3 aligned 4-Hz HR in BPM
#   - no additional interpolation is performed
#   - no paper-specific 0 / +/-2 SD cleaning is imposed because Step 2/3 already
#     produced cleaned, corrected, uniformly sampled HR; only finite positive HR
#     values are retained
#
# Outputs:
# processed_ecg_hr/paper_based_within_person_reliability_EG/
#   tables/
#   plots/
#   README_03b_reliability_outputs_EG.txt
#
# Required R packages:
#   readr, dplyr, lmerTest
#
# Install once if needed:
# install.packages(c("readr", "dplyr", "lmerTest"))

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lmerTest)
})

SCRIPT_VERSION <- "v1_tabletask_paper_based_within_person_reliability_EG"
DEFAULT_INPUT_DIR <- "processed_ecg_hr/trial_segments_4hz/paired_trial_hr"
DEFAULT_OUTPUT_DIR <- "processed_ecg_hr/paper_based_within_person_reliability_EG"
DEFAULT_ITERATIONS <- 1000L
DEFAULT_SEED <- 448L
MIN_SAMPLES_PER_HALF <- 2L

parse_args <- function(args) {
  out <- list(
    root = ".",
    input_dir = DEFAULT_INPUT_DIR,
    output_dir = DEFAULT_OUTPUT_DIR,
    random_iterations = DEFAULT_ITERATIONS,
    seed = DEFAULT_SEED,
    overwrite = FALSE
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--root") {
      i <- i + 1L; out$root <- args[[i]]
    } else if (key == "--input-dir") {
      i <- i + 1L; out$input_dir <- args[[i]]
    } else if (key == "--output-dir") {
      i <- i + 1L; out$output_dir <- args[[i]]
    } else if (key == "--random-iterations") {
      i <- i + 1L; out$random_iterations <- as.integer(args[[i]])
    } else if (key == "--seed") {
      i <- i + 1L; out$seed <- as.integer(args[[i]])
    } else if (key == "--overwrite") {
      out$overwrite <- TRUE
    } else {
      stop("Unknown argument: ", key)
    }
    i <- i + 1L
  }
  if (is.na(out$random_iterations) || out$random_iterations < 0L) {
    stop("--random-iterations must be a non-negative integer.")
  }
  out
}

resolve_path <- function(root, p) {
  if (grepl("^/", p)) normalizePath(p, mustWork = FALSE)
  else normalizePath(file.path(root, p), mustWork = FALSE)
}

first_value <- function(df, candidates, default = NA) {
  for (nm in candidates) {
    if (nm %in% names(df)) {
      x <- df[[nm]][1]
      if (!is.na(x) && trimws(as.character(x)) != "") return(x)
    }
  }
  default
}

to_bool <- function(x) {
  if (length(x) == 0L || is.na(x)) return(FALSE)
  tolower(trimws(as.character(x))) %in% c("true", "1", "yes", "y", "t")
}

read_paired_file_long <- function(path) {
  dat <- suppressMessages(read_csv(path, show_col_types = FALSE, progress = FALSE))

  required <- c("TimeRelTrialSec", "HR_A_BPM", "HR_B_BPM")
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0L) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  dat <- dat %>%
    mutate(
      TimeRelTrialSec = as.numeric(TimeRelTrialSec),
      HR_A_BPM = as.numeric(HR_A_BPM),
      HR_B_BPM = as.numeric(HR_B_BPM)
    ) %>%
    arrange(TimeRelTrialSec)

  participant_A <- first_value(dat, c("participant_A", "participant_a"))
  participant_B <- first_value(dat, c("participant_B", "participant_b"))

  # Sensor IDs are an acceptable technical fallback only when participant IDs
  # are absent; A/B role labels alone are not acceptable across files.
  if (is.na(participant_A)) participant_A <- first_value(dat, c("sensor_A", "sensor_a"))
  if (is.na(participant_B)) participant_B <- first_value(dat, c("sensor_B", "sensor_b"))
  if (is.na(participant_A) || is.na(participant_B)) {
    stop("Participant A/B identifiers are missing.")
  }

  recording_folder <- as.character(first_value(dat, c("recording_folder"), tools::file_path_sans_ext(basename(path))))
  dyad_id <- as.character(first_value(dat, c("dyad_id", "pair1"), NA))
  candidate_window <- as.character(first_value(dat, c("candidate_window"), NA))
  trial <- as.character(first_value(dat, c("trial"), NA))
  condition <- as.character(first_value(dat, c("condition"), NA))
  is_practice <- to_bool(first_value(dat, c("is_practice"), FALSE))
  is_pilot <- to_bool(first_value(dat, c("is_pilot"), FALSE))
  exclude_main <- to_bool(first_value(dat, c("exclude_from_main_analysis"), FALSE))
  exclude_reason <- as.character(first_value(dat, c("exclude_reason"), ""))

  situation_id <- paste(recording_folder, candidate_window, trial, sep = "__")

  common <- tibble(
    source_file = basename(path),
    source_path = normalizePath(path, mustWork = FALSE),
    recording_folder = recording_folder,
    dyad_id = dyad_id,
    situation_id = situation_id,
    candidate_window = candidate_window,
    trial = trial,
    condition = condition,
    is_practice = is_practice,
    is_pilot = is_pilot,
    exclude_from_main_analysis = exclude_main,
    exclude_reason = exclude_reason,
    time_s = dat$TimeRelTrialSec
  )

  bind_rows(
    common %>% mutate(participant = as.character(participant_A), role = "A", x = dat$HR_A_BPM),
    common %>% mutate(participant = as.character(participant_B), role = "B", x = dat$HR_B_BPM)
  ) %>%
    filter(is.finite(time_s), is.finite(x), x > 0) %>%
    arrange(participant, situation_id, time_s)
}

make_time_sensitive_halves <- function(df) {
  df %>%
    arrange(participant, situation_id, time_s) %>%
    group_by(
      participant, situation_id, recording_folder, dyad_id,
      candidate_window, trial, condition, is_practice, is_pilot,
      exclude_from_main_analysis, exclude_reason
    ) %>%
    summarise(
      n_samples = n(),
      n_odd = sum(seq_len(n()) %% 2L == 1L),
      n_even = sum(seq_len(n()) %% 2L == 0L),
      odd_half_mean = mean(x[seq_len(n()) %% 2L == 1L]),
      even_half_mean = mean(x[seq_len(n()) %% 2L == 0L]),
      .groups = "drop"
    ) %>%
    filter(
      n_odd >= MIN_SAMPLES_PER_HALF,
      n_even >= MIN_SAMPLES_PER_HALF,
      is.finite(odd_half_mean),
      is.finite(even_half_mean)
    )
}

make_random_halves <- function(df) {
  df %>%
    group_by(
      participant, situation_id, recording_folder, dyad_id,
      candidate_window, trial, condition, is_practice, is_pilot,
      exclude_from_main_analysis, exclude_reason
    ) %>%
    group_modify(~{
      n <- nrow(.x)
      idx <- sample.int(n, size = n, replace = FALSE)
      odd_idx <- idx[seq_along(idx) %% 2L == 1L]
      even_idx <- idx[seq_along(idx) %% 2L == 0L]
      tibble(
        n_samples = n,
        n_odd = length(odd_idx),
        n_even = length(even_idx),
        odd_half_mean = mean(.x$x[odd_idx]),
        even_half_mean = mean(.x$x[even_idx])
      )
    }) %>%
    ungroup() %>%
    filter(
      n_odd >= MIN_SAMPLES_PER_HALF,
      n_even >= MIN_SAMPLES_PER_HALF,
      is.finite(odd_half_mean),
      is.finite(even_half_mean)
    )
}

fit_paper_model <- function(split_df) {
  if (nrow(split_df) < 4L) stop("Too few participant-situation rows for mixed model.")
  if (n_distinct(split_df$participant) < 2L) stop("At least two participants are required.")
  if (sd(split_df$even_half_mean) == 0) stop("Predictor has zero variance.")

  model <- lmer(
    odd_half_mean ~ even_half_mean + (1 | participant),
    data = split_df,
    REML = TRUE
  )
  coef_tab <- coef(summary(model))
  if (!"even_half_mean" %in% rownames(coef_tab)) {
    stop("Fixed-effect slope was not estimable.")
  }

  row <- coef_tab["even_half_mean", , drop = FALSE]
  beta <- unname(row[1, "Estimate"])
  se <- unname(row[1, "Std. Error"])
  df_satt <- if ("df" %in% colnames(row)) unname(row[1, "df"]) else NA_real_
  t_value <- if ("t value" %in% colnames(row)) unname(row[1, "t value"]) else beta / se
  p_value <- if ("Pr(>|t|)" %in% colnames(row)) unname(row[1, "Pr(>|t|)"]) else NA_real_

  # Normal-approximation CI; model test/df/p-value are supplied by lmerTest.
  ci_low <- beta - qnorm(0.975) * se
  ci_high <- beta + qnorm(0.975) * se

  list(
    model = model,
    result = tibble(
      beta = beta,
      std_error = se,
      df_satterthwaite = df_satt,
      t_value = t_value,
      p_value = p_value,
      ci95_low_wald = ci_low,
      ci95_high_wald = ci_high,
      n_participant_situations = nrow(split_df),
      n_participants = n_distinct(split_df$participant),
      n_situations = n_distinct(split_df$situation_id),
      singular_fit = lme4::isSingular(model, tol = 1e-4)
    )
  )
}

analysis_strata <- function(long_df) {
  list(
    all_available = long_df,
    experimental_all_available = long_df %>% filter(!is_practice),
    main_analysis_eligible = long_df %>% filter(!is_practice, !is_pilot, !exclude_from_main_analysis),
    main_FF = long_df %>% filter(!is_practice, !is_pilot, !exclude_from_main_analysis, condition == "FF"),
    main_FE = long_df %>% filter(!is_practice, !is_pilot, !exclude_from_main_analysis, condition == "FE"),
    practice_all_available = long_df %>% filter(is_practice)
  )
}

safe_fit_stratum <- function(name, df, iterations, seed) {
  if (nrow(df) == 0L) {
    return(list(
      summary = tibble(
        stratum = name, status = "skipped", error = "No observations",
        beta = NA_real_, std_error = NA_real_, df_satterthwaite = NA_real_,
        t_value = NA_real_, p_value = NA_real_, ci95_low_wald = NA_real_,
        ci95_high_wald = NA_real_, n_participant_situations = 0L,
        n_participants = 0L, n_situations = 0L, singular_fit = NA,
        random_iterations_requested = iterations,
        random_iterations_successful = 0L,
        random_beta_mean = NA_real_, random_beta_sd = NA_real_,
        random_beta_median = NA_real_, random_beta_q025 = NA_real_,
        random_beta_q975 = NA_real_
      ),
      time_halves = tibble(), random_betas = tibble(), model_text = ""
    ))
  }

  time_halves <- make_time_sensitive_halves(df)

  time_fit <- tryCatch(
    fit_paper_model(time_halves),
    error = function(e) e
  )

  if (inherits(time_fit, "error")) {
    return(list(
      summary = tibble(
        stratum = name, status = "failed", error = conditionMessage(time_fit),
        beta = NA_real_, std_error = NA_real_, df_satterthwaite = NA_real_,
        t_value = NA_real_, p_value = NA_real_, ci95_low_wald = NA_real_,
        ci95_high_wald = NA_real_,
        n_participant_situations = nrow(time_halves),
        n_participants = n_distinct(time_halves$participant),
        n_situations = n_distinct(time_halves$situation_id),
        singular_fit = NA,
        random_iterations_requested = iterations,
        random_iterations_successful = 0L,
        random_beta_mean = NA_real_, random_beta_sd = NA_real_,
        random_beta_median = NA_real_, random_beta_q025 = NA_real_,
        random_beta_q975 = NA_real_
      ),
      time_halves = time_halves, random_betas = tibble(), model_text = ""
    ))
  }

  set.seed(seed)
  random_beta <- rep(NA_real_, iterations)
  if (iterations > 0L) {
    for (i in seq_len(iterations)) {
      random_halves <- make_random_halves(df)
      fit_i <- tryCatch(fit_paper_model(random_halves), error = function(e) NULL)
      if (!is.null(fit_i)) random_beta[i] <- fit_i$result$beta
    }
  }
  random_tbl <- tibble(
    iteration = seq_len(iterations),
    beta = random_beta,
    successful = is.finite(random_beta)
  )
  good <- random_beta[is.finite(random_beta)]

  result <- time_fit$result %>%
    mutate(
      stratum = name,
      status = "success",
      error = "",
      random_iterations_requested = iterations,
      random_iterations_successful = length(good),
      random_beta_mean = if (length(good)) mean(good) else NA_real_,
      random_beta_sd = if (length(good) > 1L) sd(good) else NA_real_,
      random_beta_median = if (length(good)) median(good) else NA_real_,
      random_beta_q025 = if (length(good)) unname(quantile(good, 0.025)) else NA_real_,
      random_beta_q975 = if (length(good)) unname(quantile(good, 0.975)) else NA_real_
    ) %>%
    select(
      stratum, status, error, everything()
    )

  model_text <- paste(capture.output(summary(time_fit$model)), collapse = "\n")

  list(
    summary = result,
    time_halves = time_halves,
    random_betas = random_tbl,
    model_text = model_text
  )
}

save_histogram <- function(beta, time_beta, title, path) {
  beta <- beta[is.finite(beta)]
  if (length(beta) == 0L) return(invisible(FALSE))
  png(path, width = 1800, height = 1200, res = 180)
  hist(
    beta,
    breaks = 40,
    main = title,
    xlab = "Mixed-model fixed-effect slope (beta)",
    border = "black",
    col = "white"
  )
  abline(v = time_beta, lwd = 2, lty = 2)
  dev.off()
  invisible(TRUE)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- normalizePath(args$root, mustWork = TRUE)
  input_dir <- resolve_path(root, args$input_dir)
  output_dir <- resolve_path(root, args$output_dir)

  if (!dir.exists(input_dir)) stop("Input directory not found: ", input_dir)
  if (dir.exists(output_dir) && args$overwrite) unlink(output_dir, recursive = TRUE, force = TRUE)

  table_dir <- file.path(output_dir, "tables")
  plot_dir <- file.path(output_dir, "plots")
  dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  files <- sort(list.files(input_dir, pattern = "\\.csv$", full.names = TRUE))
  files <- files[!grepl("/\\._", files)]
  if (length(files) == 0L) stop("No paired Step 3 CSV files found.")

  cat("Script version:", SCRIPT_VERSION, "\n")
  cat("Paired files found:", length(files), "\n")

  long_parts <- vector("list", length(files))
  manifest <- vector("list", length(files))
  for (i in seq_along(files)) {
    res <- tryCatch(read_paired_file_long(files[[i]]), error = function(e) e)
    if (inherits(res, "error")) {
      manifest[[i]] <- tibble(
        source_file = basename(files[[i]]),
        status = "failed",
        n_long_rows = 0L,
        error = conditionMessage(res)
      )
    } else {
      long_parts[[i]] <- res
      manifest[[i]] <- tibble(
        source_file = basename(files[[i]]),
        status = "success",
        n_long_rows = nrow(res),
        error = ""
      )
    }
    if (i %% 10L == 0L || i == length(files)) {
      cat("  read", i, "/", length(files), "files\n")
    }
  }

  manifest_df <- bind_rows(manifest)
  write_csv(manifest_df, file.path(table_dir, "03b_reliability_file_manifest_EG.csv"))

  long_df <- bind_rows(long_parts)
  if (nrow(long_df) == 0L) stop("No usable HR observations were loaded.")
  write_csv(long_df, file.path(table_dir, "03b_reliability_long_input_EG.csv"))

  strata <- analysis_strata(long_df)
  results <- list()

  for (i in seq_along(strata)) {
    nm <- names(strata)[[i]]
    cat("Analyzing stratum:", nm, "\n")
    out <- safe_fit_stratum(
      name = nm,
      df = strata[[i]],
      iterations = args$random_iterations,
      seed = args$seed + i
    )
    results[[nm]] <- out

    if (nrow(out$time_halves) > 0L) {
      write_csv(
        out$time_halves,
        file.path(table_dir, paste0("03b_time_sensitive_split_halves_", nm, "_EG.csv"))
      )
    }
    if (nrow(out$random_betas) > 0L) {
      write_csv(
        out$random_betas,
        file.path(table_dir, paste0("03b_random_split_betas_", nm, "_EG.csv"))
      )
    }
    if (nzchar(out$model_text)) {
      writeLines(
        out$model_text,
        file.path(table_dir, paste0("03b_time_sensitive_model_", nm, "_EG.txt"))
      )
    }
    if (nrow(out$random_betas) > 0L && out$summary$status[[1]] == "success") {
      save_histogram(
        out$random_betas$beta,
        out$summary$beta[[1]],
        paste("Random split-half beta distribution:", nm),
        file.path(plot_dir, paste0("03b_random_split_beta_distribution_", nm, "_EG.png"))
      )
    }
  }

  summary_df <- bind_rows(lapply(results, `[[`, "summary"))
  write_csv(summary_df, file.path(table_dir, "03b_within_person_reliability_summary_EG.csv"))

  run_summary <- c(
    "TableTask paper-based within-participant reliability",
    "====================================================",
    paste0("script_version: ", SCRIPT_VERSION),
    paste0("root: ", root),
    paste0("input_dir: ", input_dir),
    paste0("output_dir: ", output_dir),
    paste0("n_paired_files_found: ", length(files)),
    paste0("n_files_successfully_loaded: ", sum(manifest_df$status == "success")),
    paste0("n_files_failed: ", sum(manifest_df$status == "failed")),
    paste0("n_long_hr_rows: ", nrow(long_df)),
    paste0("n_unique_participants: ", n_distinct(long_df$participant)),
    paste0("n_unique_situations: ", n_distinct(long_df$situation_id)),
    paste0("random_split_iterations: ", args$random_iterations),
    paste0("random_seed: ", args$seed),
    "",
    "Paper-consistent model:",
    "odd_half_mean ~ even_half_mean + (1 | participant)",
    "",
    "Interpretation:",
    "The fixed-effect slope beta is the paper-style within-participant reliability estimate.",
    "The time-sensitive estimate uses chronologically ordered odd/even samples.",
    "The random split sensitivity analysis repeats random half-splitting 1000 times by default.",
    "",
    "Scope:",
    "This is a study-level reliability analysis and does not replace trial-level Step 3a QC."
  )
  writeLines(run_summary, file.path(table_dir, "03b_reliability_run_summary_EG.txt"))

  readme <- c(
    "TableTask paper-based within-participant reliability outputs",
    "===========================================================",
    "",
    "Primary result:",
    "tables/03b_within_person_reliability_summary_EG.csv",
    "",
    "Method:",
    "Chronological odd/even split within participant x trial, followed by",
    "odd_half_mean ~ even_half_mean + (1 | participant).",
    "",
    "The random split-half sensitivity analysis repeats the split and mixed model",
    "1000 times by default and reports the distribution of fixed-effect beta values.",
    "",
    "This script is separate from Step 3a odd-even RR consistency QC."
  )
  writeLines(readme, file.path(output_dir, "README_03b_reliability_outputs_EG.txt"))

  cat("\nDone.\n")
  cat("Primary result:", file.path(table_dir, "03b_within_person_reliability_summary_EG.csv"), "\n")
}

main()
