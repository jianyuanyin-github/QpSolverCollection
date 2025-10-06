/* Author: Masaki Murooka */

#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_PROXQP
#include <qp_solver_collection/QpSolverCollection.h>

#include <proxsuite/proxqp/dense/dense.hpp>

using namespace QpSolverCollection;
QpSolverProxqp::QpSolverProxqp()
{
  type_ = QpSolverType::PROXQP;

  // Initialize PROXQP parameters
  declare_and_update_parameters();
}
void QpSolverProxqp::declare_and_update_parameters()
{
  proxqp_params_.eps_abs =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.eps_abs", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  proxqp_params_.eps_rel =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.eps_rel", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  proxqp_params_.max_iter =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.max_iter", 30, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  proxqp_params_.verbose =
    param_manager_
      ->declare_and_get_value("MPC.Solver_PROXQP.verbose", false, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  proxqp_params_.warm_start =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.warm_start", true, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  proxqp_params_.compute_timings =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.compute_timings", false, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  proxqp_params_.check_duality_gap =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.check_duality_gap", false, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  previous_param_state_hash_ = param_manager_->get_state_hash();
}
Eigen::VectorXd QpSolverProxqp::solve(
  int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
  const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
  const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
  const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
  const Eigen::Ref<const Eigen::VectorXd> & x_max)
{
  // Check if parameters have changed and update if necessary
  if (param_manager_->get_state_hash() != previous_param_state_hash_) {
    declare_and_update_parameters();
  }

  // Only recreate solver if dimensions changed or if not initialized
  int dim_ineq_with_bound = dim_ineq + dim_var;
  if (!proxqp_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
    proxqp_ =
      std::make_unique<proxsuite::proxqp::dense::QP<double>>(dim_var, dim_eq, dim_ineq_with_bound);
    // Store dimensions for incremental updates
    dim_var_ = dim_var;
    dim_eq_ = dim_eq;
    dim_ineq_ = dim_ineq;
  }

  proxqp_->settings.eps_abs = proxqp_params_.eps_abs;
  proxqp_->settings.eps_rel = proxqp_params_.eps_rel;
  proxqp_->settings.max_iter = proxqp_params_.max_iter;
  proxqp_->settings.verbose = proxqp_params_.verbose;
  proxqp_->settings.compute_timings = proxqp_params_.compute_timings;
  proxqp_->settings.check_duality_gap = proxqp_params_.check_duality_gap;
  proxqp_->settings.initial_guess =
    proxqp_params_.warm_start
      ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
      : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;

  Eigen::MatrixXd C_with_bound(dim_ineq_with_bound, dim_var);
  Eigen::VectorXd d_with_bound_min(dim_ineq_with_bound);
  Eigen::VectorXd d_with_bound_max(dim_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);

  // Correctly stack matrices row-wise (not column-wise)
  C_with_bound.topRows(dim_ineq) = C;
  C_with_bound.bottomRows(dim_var) = I;

  // Set bounds: inequality constraints (C x <= d) and variable bounds
  d_with_bound_min.head(dim_ineq) = Eigen::VectorXd::Constant(dim_ineq, -1e30);
  d_with_bound_min.tail(dim_var) = x_min;
  d_with_bound_max.head(dim_ineq) = d;
  d_with_bound_max.tail(dim_var) = x_max;

  proxqp_->update(Q, c, A, b, C_with_bound, d_with_bound_min, d_with_bound_max);
  auto solve_start_time = clock::now();
  proxqp_->solve();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    logger_->log("proxqp_iter", proxqp_->results.info.iter);
    logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
    logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
  }

  if (proxqp_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM(
      "[QpSolverProxqp::solve] Failed to solve: "
      << static_cast<int>(proxqp_->results.info.status));
  }

  return proxqp_->results.x;
}
Eigen::VectorXd QpSolverProxqp::solve(
  int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
  const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
  const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
  const Eigen::Ref<const Eigen::VectorXd> & d_lower,
  const Eigen::Ref<const Eigen::VectorXd> & d_upper,
  const Eigen::Ref<const Eigen::VectorXd> & x_min, const Eigen::Ref<const Eigen::VectorXd> & x_max)
{
  // Check if parameters have changed and update if necessary
  if (param_manager_->get_state_hash() != previous_param_state_hash_) {
    declare_and_update_parameters();
  }

  // Only recreate solver if dimensions changed or if not initialized
  // ProxQP constructor: QP(n_vars, n_equality, n_inequality)
  // For bilateral constraints, we combine equality and inequality into one constraint block
  int n_inequality = dim_ineq + dim_var;  // inequality constraints + variable bounds
  if (!proxqp_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
    proxqp_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(dim_var, dim_eq, n_inequality);
    // Store dimensions for incremental updates
    dim_var_ = dim_var;
    dim_eq_ = dim_eq;
    dim_ineq_ = dim_ineq;
  }

  proxqp_->settings.eps_abs = proxqp_params_.eps_abs;
  proxqp_->settings.eps_rel = proxqp_params_.eps_rel;
  proxqp_->settings.max_iter = proxqp_params_.max_iter;
  proxqp_->settings.verbose = proxqp_params_.verbose;
  proxqp_->settings.initial_guess =
    proxqp_params_.warm_start
      ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
      : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  proxqp_->settings.compute_timings = proxqp_params_.compute_timings;
  proxqp_->settings.check_duality_gap = proxqp_params_.check_duality_gap;

  // Build constraint matrices: A for equality, C_with_bound for inequality
  Eigen::MatrixXd C_with_bound(n_inequality, dim_var);
  Eigen::VectorXd l_with_bound(n_inequality);
  Eigen::VectorXd u_with_bound(n_inequality);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);

  // Correctly stack inequality matrices row-wise
  C_with_bound.topRows(dim_ineq) = C;
  C_with_bound.bottomRows(dim_var) = I;

  // Use bilateral constraints: d_lower <= Cx <= d_upper
  l_with_bound.head(dim_ineq) = d_lower;
  l_with_bound.tail(dim_var) = x_min;
  u_with_bound.head(dim_ineq) = d_upper;
  u_with_bound.tail(dim_var) = x_max;

  proxqp_->update(Q, c, A, b, C_with_bound, l_with_bound, u_with_bound);
  auto solve_start_time = clock::now();
  proxqp_->solve();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    logger_->log("proxqp_iter", proxqp_->results.info.iter);
    logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
    logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
  }

  if (proxqp_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM(
      "[QpSolverProxqp::solve bilateral] Failed to solve: "
      << static_cast<int>(proxqp_->results.info.status));
  }
  return proxqp_->results.x;
}
// ===== INCREMENTAL UPDATE IMPLEMENTATION =====
bool QpSolverProxqp::updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateObjectiveMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (Q.rows() != dim_var_ || Q.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateObjectiveMatrix] Dimension mismatch");
    return false;
  }

  proxqp_->update(
    Q, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateObjectiveVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (c.size() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateObjectiveVector] Dimension mismatch");
    return false;
  }

  proxqp_->update(
    std::nullopt, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (C.rows() != dim_ineq_ || C.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityMatrix] Dimension mismatch");
    return false;
  }

  if (dim_eq_ > 0) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityMatrix] Cannot update inequality matrix when equality "
      "constraints exist. Use full solve() instead.");
    return false;
  }

  // Build constraint matrix with variable bounds
  int n_inequality = dim_ineq_ + dim_var_;
  Eigen::MatrixXd C_with_bound(n_inequality, dim_var_);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var_, dim_var_);

  C_with_bound.topRows(dim_ineq_) = C;
  C_with_bound.bottomRows(dim_var_) = I;

  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, std::nullopt, C_with_bound, std::nullopt,
    std::nullopt, std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (d.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityVector] Dimension mismatch");
    return false;
  }

  if (dim_eq_ > 0) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVector] Cannot update inequality vector when equality "
      "constraints exist. Use full solve() instead.");
    return false;
  }

  // Build bounds vectors: only update inequality part, keep variable bounds unchanged
  // Note: This assumes variable bounds don't change. If they do, use full solve().
  int n_inequality = dim_ineq_ + dim_var_;
  Eigen::VectorXd l_with_bound(n_inequality);
  Eigen::VectorXd u_with_bound(n_inequality);

  // Set inequality constraints (C x <= d, so lower bound is -1e30)
  l_with_bound.head(dim_ineq_) = Eigen::VectorXd::Constant(dim_ineq_, -1e30);
  u_with_bound.head(dim_ineq_) = d;

  // Keep variable bounds unchanged (ProxQP will preserve existing bounds via std::nullopt
  // semantics) We pass nullopt to avoid changing the variable bound part Actually, we need to pass
  // full vectors, so we use large values
  l_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, -1e30);
  u_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, 1e30);

  QSC_WARN_STREAM(
    "[QpSolverProxqp::updateInequalityVector] Warning: Variable bounds reset to default. Use full "
    "solve() to preserve custom bounds.");

  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, l_with_bound,
    u_with_bound, std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateInequalityVectorBothSide(
  const Eigen::Ref<const Eigen::VectorXd> & d_lower,
  const Eigen::Ref<const Eigen::VectorXd> & d_upper)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVectorBothSide] Solver not initialized. Call solve() "
      "first.");
    return false;
  }

  if (d_lower.size() != dim_ineq_ || d_upper.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityVectorBothSide] Dimension mismatch");
    return false;
  }

  if (dim_eq_ > 0) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVectorBothSide] Cannot update with equality constraints. "
      "Use full solve() instead.");
    return false;
  }

  int n_inequality = dim_ineq_ + dim_var_;
  Eigen::VectorXd l_with_bound(n_inequality);
  Eigen::VectorXd u_with_bound(n_inequality);

  // Set bilateral inequality constraints
  l_with_bound.head(dim_ineq_) = d_lower;
  u_with_bound.head(dim_ineq_) = d_upper;

  // Variable bounds - reset to default (limitation without caching)
  l_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, -1e30);
  u_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, 1e30);

  QSC_WARN_STREAM(
    "[QpSolverProxqp::updateInequalityVectorBothSide] Warning: Variable bounds reset to default. "
    "Use full solve() to preserve custom bounds.");

  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, l_with_bound,
    u_with_bound, std::nullopt, std::nullopt);

  return true;
}
bool QpSolverProxqp::updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateEqualityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (A.rows() != dim_eq_ || A.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateEqualityMatrix] Dimension mismatch");
    return false;
  }

  proxqp_->update(
    std::nullopt, std::nullopt, A, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b)
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateEqualityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (b.size() != dim_eq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateEqualityVector] Dimension mismatch");
    return false;
  }

  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, b, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
Eigen::VectorXd QpSolverProxqp::solveIncremental()
{
  if (!proxqp_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::solveIncremental] Solver not initialized. Call solve() first.");
    return Eigen::VectorXd::Zero(0);
  }

  auto solve_start_time = clock::now();
  proxqp_->solve();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    logger_->log("proxqp_iter", proxqp_->results.info.iter);
    logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
    logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
  }

  if (proxqp_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM(
      "[QpSolverProxqp::solveIncremental] Failed to solve: "
      << static_cast<int>(proxqp_->results.info.status));
  }

  return proxqp_->results.x;
}
namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverProxqp() { return std::make_shared<QpSolverProxqp>(); }
}  // namespace QpSolverCollection
#endif
