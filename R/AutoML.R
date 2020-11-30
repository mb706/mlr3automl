#' @title Interface function for mlr3automl
#'
#' @description
#' Creates an instance of [AutoMLClassif][mlr3automl::AutoMLClassif] or [AutoMLRegr][mlr3automl::AutoMLRegr].
#'
#' @param task ([`Task`][mlr3::Task]) \cr
#' Contains the dataset for which to create a model. Currently [`TaskClassif`][mlr3::TaskClassif] and [`TaskRegr`][mlr3::TaskRegr] are supported.
#' @param learner_list (`list()` | `character()`) \cr  # TODO_MB: why do we allow list *and* character here?
#' `List` of names from [mlr_learners][mlr3::mlr_learners]. Can be used to customize the learners to be tuned over. \cr
#' Default learners for classification: `c("classif.ranger", "classif.xgboost", "classif.liblinear")` \cr
#' Default learners for regression: `c("regr.ranger", "regr.xgboost", "regr.svm", "regr.liblinear", "regr.cv_glmnet")` \cr
#' Might break mlr3automl if a user-provided learner is incompatible with the provided task.  # TODO_MB: "break mlr3automl" should not happen. Instead, throw an error if a given learner is incompatible.
#' @param learner_timeout (`integer(1)`) \cr
#' Budget (in seconds) for a single parameter evaluation during model training. \cr  # Be more precise, Is this training and prediction timeout combined? Timeout for one whole resampling-run (even when it is crossvalidation)?
#' If this budget is exceeded, the evaluation is stopped and performance measured with the fallback
#' [LearnerClassifFeatureless][mlr3::LearnerClassifFeatureless] or [LearnerRegrFeatureless][mlr3::LearnerRegrFeatureless]. \cr
#' When this is `NULL` (default), it defaults to `runtime / 5`.  # TODO_MB: why not set this to `runtime / 5` directly, then? See my comment below
#' @param resampling ([Resampling][mlr3::Resampling]) \cr
#' The resampling method to use for performance evaluation.
#' Defaults to [ResamplingHoldout][mlr3::ResamplingHoldout].
#' @param measure ([Measure][mlr3::Measure]) \cr
#' Performance measure for which to optimize during training. \cr
#' Defaults to [Accuracy][mlr3measures::acc] for classification and [RMSE][mlr3measures::acc] for regression.
#' @param runtime (`integer(1)`) \cr  # TODO_MB: why do we insist on integer seconds here?
#' Number of seconds for which to run the optimization. Does *not* include training time of the final model. \cr
#' Defaults to `Inf`, letting [Hyperband][mlr3hyperband] terminate the tuning.
#' @param terminator ([Terminator][bbotk::Terminator]) \cr
#' Optional additional termination criterion for model tuning, besides runtime. \cr
#' Note that the [Hyperband][mlr3hyperband] tuner might stop training before the budget is exhausted.
#' [TerminatorRunTime][bbotk::TerminatorRunTime] should not be used, use the separate `runtime` parameter instead. \cr
#' Defaults to [TerminatorNone][bbotk::TerminatorNone], i.e. run for `runtime` seconds or until the tuner terminates by itself.
#' @param preprocessing (`character(1)` | [Graph][mlr3pipelines::Graph]) \cr
#' Type of preprocessing to be used. Possible values are :
#' - "none": No preprocessing at all
#' - "stability": [`pipeline_robustify`][mlr3pipelines::pipeline_robustify] is used to guarantee stability of the learners in the pipeline  # TODO_MB: maybe use a better description here. E.g. 'Use simple preprocessing with the aim of making the evaluation process more robust, while modifying the data as little as possible'
#' - "full": Adds additional preprocessing operators for [Imputation][mlr3pipelines::PipeOpImpute], [Impact Encoding][mlr3pipelines::PipeOpEncodeImpact] and [PCA][mlr3pipelines::PipeOpPCA]. \cr
#' The choice of preprocessing operators is optimised during tuning.  # TODO_MB: this sounds like the choice between "none", "stability" , or "full" is optimized, but what you mean is that the hyperparameters of the tuning pipeline are optimized, right?
#'
#' Alternatively, a [Graph][mlr3pipelines::Graph] object can be used to specify a custom preprocessing pipeline.
#' @param portfolio (`logical(1)`) \cr
#' `mlr3automl` tries out a fixed portfolio of known good learners prior to tuning. \cr
#' The `portfolio` parameter disables trying these portfolio learners.  # TODO_MB: this sounds like setting `portfolio` to `TRUE` *disables* something. Instead, write something like 'whether to try out a fixed portfolio... defaults to TRUE'.
#'
#' @return ([AutoMLClassif][mlr3automl::AutoMLClassif] | [AutoMLRegr][mlr3automl::AutoMLRegr]) \cr
#' Returned class corresponds to the type of `task`.
#' @export
#'
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3automl)
#'
#' model = AutoML(tsk("iris"))
#' model$train()
#' }
# TODO_MB: it would be good to have the descriptions of '@param's that are not in the first line start with spaces. Example:
# @param parameter (`logical(1)`)\cr
#   This is the description. It starts with two spaces of indentation to make it visible that it belongs to the above @param.
# @param another_parameter (`NULL`)\cr
#   Contains nothing.
# TODO_MB: just have learner_timeout = runtime / 5.
# TODO_MB: think about the ordering of parameters. I would say `runtime` is more fundamental than `learner_timeout` and their order should be swapped.
AutoML = function(task, learner_list = NULL, learner_timeout = NULL,
                  resampling = NULL, measure = NULL, runtime = Inf,
                  terminator = NULL, preprocessing = NULL,
                  portfolio = TRUE) {
  assert_r6(task, "Task")
  if (task$task_type == "classif") {
    # stratify target variable so that every target label appears
    # in all folds while resampling
    # TODO_MB: I don't get why you do this, are there task_type == "classif" tasks that don't have this? (And even if there are, why wouldn't we stratify by target for them?)
    target_is_factor = task$col_info[task$col_info$id == task$target_names, ]$type == "factor"
    if (length(target_is_factor) == 1 && target_is_factor) {
      task$col_roles$stratum = task$target_names
    }
    # TODO_MB: when there are this many arguments I would consider it good practice to name the arguments, e.g. `$new(task = task, learner_list = learner_list, ...`
    return(AutoMLClassif$new(task, learner_list, learner_timeout,
                             resampling, measure, runtime, terminator, preprocessing, portfolio))
  } else if (task$task_type == "regr") {
    return(AutoMLRegr$new(task, learner_list, learner_timeout,
                          resampling, measure, runtime, terminator, preprocessing, portfolio))
  } else {
    stop("mlr3automl only supports classification and regression tasks for now")
  }
}
