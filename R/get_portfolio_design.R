get_portfolio_design = function(task_type, param_set, learner_list) {
  initial_design = data.table::data.table(
        subsample.frac = c(0.1, 1, 0.1, 0.33, 1, 1, 1),
        stability.missind.type = "numeric",
        numeric.branch.selection = "imputation.imputemean",
        factor.branch.selection = "imputation.imputeoor",
        encoding.branch.selection = c("stability.encodeimpact", "stability.encodeimpact", NA_character_, NA_character_, "stability.encodeimpact", "stability.encodeimpact", NA_character_),
        dimensionality.branch.selection = "dimensionality.nop",
        branch.selection = paste0(task_type, c(".featureless", ".liblinear", ".ranger", ".ranger", ".xgboost", ".xgboost", ".ranger")),
        dimensionality.pca.rank. = NA_integer_,
        classif.ranger.mtry = c(NA_real_, NA_real_, 0.5, 0.5, NA_real_, NA_real_, 0.3),
        classif.xgboost.booster = c(NA_character_, NA_character_, NA_character_, NA_character_, "gbtree", "gbtree", NA_character_),
        classif.xgboost.sample_type = NA_character_,
        classif.xgboost.normalize_type = NA_character_,
        classif.xgboost.rate_drop = NA_real_,
        classif.xgboost.eta = NA_real_,
        classif.xgboost.nrounds = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_, 100, 100, NA_integer_),
        classif.xgboost.alpha = NA_real_,
        classif.xgboost.lambda = NA_real_,
        classif.xgboost.subsample = c(NA_real_, NA_real_, NA_real_, NA_real_, 0.1, 0.33, NA_real_),
        classif.xgboost.colsample_bytree = NA_real_,
        classif.xgboost.colsample_bylevel = NA_real_,
        classif.xgboost.max_depth = NA_integer_,
        classif.xgboost.min_child_weight = NA_integer_,
        classif.xgboost.gamma = NA_real_,
        classif.liblinear.cost = c(NA_real_, 1, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_),
        classif.liblinear.type = c(NA_character_, "0", NA_character_, NA_character_, NA_character_, NA_character_, NA_character_),
        classif.ranger.splitrule = NA_character_,
        classif.cv_glmnet.alpha = NA_real_,
        classif.svm.kernel = NA_character_,
        classif.svm.cost = NA_real_,
        classif.svm.gamma = NA_real_,
        classif.svm.type = NA_character_,
        regr.ranger.mtry = c(NA_real_, NA_real_, 0.5, 0.5, NA_real_, NA_real_, 0.3),
        regr.xgboost.booster = c(NA_character_, NA_character_, NA_character_, NA_character_, "gbtree", "gbtree", NA_character_),
        regr.xgboost.sample_type = NA_character_,
        regr.xgboost.normalize_type = NA_character_,
        regr.xgboost.rate_drop = NA_real_,
        regr.xgboost.eta = NA_real_,
        regr.xgboost.nrounds = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_, 100, 100, NA_integer_),
        regr.xgboost.alpha = NA_real_,
        regr.xgboost.lambda = NA_real_,
        regr.xgboost.subsample = c(NA_real_, NA_real_, NA_real_, NA_real_, 0.1, 0.33, NA_real_),
        regr.xgboost.colsample_bytree = NA_real_,
        regr.xgboost.colsample_bylevel = NA_real_,
        regr.xgboost.max_depth = NA_integer_,
        regr.xgboost.min_child_weight = NA_integer_,
        regr.xgboost.gamma = NA_real_,
        regr.liblinear.cost = c(NA_real_, 1, NA_real_, NA_real_, NA_real_, NA_real_, NA_real_),
        regr.liblinear.type = c(NA_character_, "11", NA_character_, NA_character_, NA_character_, NA_character_, NA_character_),
        regr.ranger.splitrule = NA_character_,
        regr.cv_glmnet.alpha = NA_real_,
        regr.svm.kernel = NA_character_,
        regr.svm.cost = NA_real_,
        regr.svm.gamma = NA_real_,
        regr.svm.type = NA_character_
  )
  if ("encoding.branch.selection" %in% param_set$ids()) {
    return(initial_design[initial_design$branch.selection %in% learner_list &
                          initial_design$encoding.branch.selection %in% c(param_set$levels$encoding.branch.selection, NA),
                          param_set$ids(),
                          with = FALSE])
  }

  return(initial_design[initial_design$branch.selection %in% learner_list,
                        param_set$ids(),
                        with = FALSE])
}


