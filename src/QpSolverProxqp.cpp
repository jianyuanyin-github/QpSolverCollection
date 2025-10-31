/* Author: Masaki Murooka */

#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_PROXQP
#include <qp_solver_collection/QpSolverCollection.h>

#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

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
        "MPC.Solver_PROXQP.max_iter", 60, tam::pmg::ParameterType::INTEGER, "")
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
  proxqp_params_.rho =
    param_manager_
      ->declare_and_get_value("MPC.Solver_PROXQP.rho", 1e-1, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  proxqp_params_.use_sparse =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.use_sparse", true, tam::pmg::ParameterType::BOOL, "")
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

  if (proxqp_params_.use_sparse) {
    // Sparse mode
    if (!proxqp_sparse_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
      proxqp_sparse_ = std::make_unique<proxsuite::proxqp::sparse::QP<double, int>>(
        dim_var, dim_eq, dim_ineq_with_bound);
      dim_var_ = dim_var;
      dim_eq_ = dim_eq;
      dim_ineq_ = dim_ineq;
    }
    proxqp_sparse_->settings.eps_abs = proxqp_params_.eps_abs;
    proxqp_sparse_->settings.eps_rel = proxqp_params_.eps_rel;
    proxqp_sparse_->settings.max_iter = proxqp_params_.max_iter;
    proxqp_sparse_->settings.verbose = proxqp_params_.verbose;
    proxqp_sparse_->settings.compute_timings = proxqp_params_.compute_timings;
    proxqp_sparse_->settings.check_duality_gap = proxqp_params_.check_duality_gap;
    proxqp_sparse_->settings.default_rho = proxqp_params_.rho;
    proxqp_sparse_->settings.initial_guess =
      proxqp_params_.warm_start
        ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
        : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  } else {
    // Dense mode
    if (!proxqp_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
      proxqp_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(
        dim_var, dim_eq, dim_ineq_with_bound);
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
    proxqp_->settings.default_rho = proxqp_params_.rho;
    proxqp_->settings.initial_guess =
      proxqp_params_.warm_start
        ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
        : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
  }

  Eigen::MatrixXd C_with_bound(dim_ineq_with_bound, dim_var);
  Eigen::VectorXd d_with_bound_min(dim_ineq_with_bound);
  Eigen::VectorXd d_with_bound_max(dim_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);

  // Correctly stack matrices row-wise (not column-wise)
  C_with_bound.topRows(dim_ineq) = C;
  C_with_bound.bottomRows(dim_var) = I;

  d_with_bound_min.head(dim_ineq) = Eigen::VectorXd::Constant(dim_ineq, -1e8);
  d_with_bound_min.tail(dim_var) = x_min;
  d_with_bound_max.head(dim_ineq) = d;
  d_with_bound_max.tail(dim_var) = x_max;

  auto solve_start_time = clock::now();

  if (proxqp_params_.use_sparse) {
    // Convert dense matrices to sparse with correct index type
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Q_sparse = Q.sparseView();
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_sparse = A.sparseView();
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> C_sparse = C_with_bound.sparseView();

    proxqp_sparse_->init(Q_sparse, c, A_sparse, b, C_sparse, d_with_bound_min, d_with_bound_max);
    proxqp_sparse_->solve();
  } else {
    proxqp_->update(Q, c, A, b, C_with_bound, d_with_bound_min, d_with_bound_max);
    proxqp_->solve();
  }

  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    if (proxqp_params_.use_sparse) {
      logger_->log("proxqp_iter", proxqp_sparse_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_sparse_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_sparse_->results.info.dua_res);
    } else {
      logger_->log("proxqp_iter", proxqp_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
    }
  }

  if (proxqp_params_.use_sparse) {
    auto status = proxqp_sparse_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM(
        "[QpSolverProxqp::solve sparse] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_sparse_->results.x;
  } else {
    auto status = proxqp_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM("[QpSolverProxqp::solve] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_->results.x;
  }
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

  if (proxqp_params_.use_sparse) {
    // Sparse mode
    if (!proxqp_sparse_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
      proxqp_sparse_ =
        std::make_unique<proxsuite::proxqp::sparse::QP<double, int>>(dim_var, dim_eq, n_inequality);
      dim_var_ = dim_var;
      dim_eq_ = dim_eq;
      dim_ineq_ = dim_ineq;
    }
    proxqp_sparse_->settings.eps_abs = proxqp_params_.eps_abs;
    proxqp_sparse_->settings.eps_rel = proxqp_params_.eps_rel;
    proxqp_sparse_->settings.max_iter = proxqp_params_.max_iter;
    proxqp_sparse_->settings.verbose = proxqp_params_.verbose;
    proxqp_sparse_->settings.default_rho = proxqp_params_.rho;
    proxqp_sparse_->settings.initial_guess =
      proxqp_params_.warm_start
        ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
        : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    proxqp_sparse_->settings.compute_timings = proxqp_params_.compute_timings;
    proxqp_sparse_->settings.check_duality_gap = proxqp_params_.check_duality_gap;
  } else {
    // Dense mode
    if (!proxqp_ || dim_var != dim_var_ || dim_eq != dim_eq_ || dim_ineq != dim_ineq_) {
      proxqp_ =
        std::make_unique<proxsuite::proxqp::dense::QP<double>>(dim_var, dim_eq, n_inequality);
      dim_var_ = dim_var;
      dim_eq_ = dim_eq;
      dim_ineq_ = dim_ineq;
    }
    proxqp_->settings.eps_abs = proxqp_params_.eps_abs;
    proxqp_->settings.eps_rel = proxqp_params_.eps_rel;
    proxqp_->settings.max_iter = proxqp_params_.max_iter;
    proxqp_->settings.verbose = proxqp_params_.verbose;
    proxqp_->settings.default_rho = proxqp_params_.rho;
    proxqp_->settings.initial_guess =
      proxqp_params_.warm_start
        ? proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT
        : proxsuite::proxqp::InitialGuessStatus::NO_INITIAL_GUESS;
    proxqp_->settings.compute_timings = proxqp_params_.compute_timings;
    proxqp_->settings.check_duality_gap = proxqp_params_.check_duality_gap;
  }

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

  auto solve_start_time = clock::now();

  if (proxqp_params_.use_sparse) {
    // Convert dense matrices to sparse with correct index type
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Q_sparse = Q.sparseView();
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_sparse = A.sparseView();
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> C_sparse = C_with_bound.sparseView();

    proxqp_sparse_->init(Q_sparse, c, A_sparse, b, C_sparse, l_with_bound, u_with_bound);
    proxqp_sparse_->solve();
  } else {
    proxqp_->update(Q, c, A, b, C_with_bound, l_with_bound, u_with_bound);
    proxqp_->solve();
  }

  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    if (proxqp_params_.use_sparse) {
      logger_->log("proxqp_iter", proxqp_sparse_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_sparse_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_sparse_->results.info.dua_res);
    } else {
      logger_->log("proxqp_iter", proxqp_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
    }
  }

  if (proxqp_params_.use_sparse) {
    auto status = proxqp_sparse_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM(
        "[QpSolverProxqp::solve bilateral sparse] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_sparse_->results.x;
  } else {
    auto status = proxqp_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM(
        "[QpSolverProxqp::solve bilateral] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_->results.x;
  }
}
// ===== INCREMENTAL UPDATE IMPLEMENTATION =====
bool QpSolverProxqp::updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateObjectiveMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (Q.rows() != dim_var_ || Q.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateObjectiveMatrix] Dimension mismatch");
    return false;
  }

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> Q_sparse = Q.sparseView();
    proxqp_sparse_->update(
      Q_sparse, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      Q, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateObjectiveVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (c.size() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateObjectiveVector] Dimension mismatch");
    return false;
  }

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    proxqp_sparse_->update(
      std::nullopt, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (C.rows() != dim_ineq_ || C.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityMatrix] Dimension mismatch");
    return false;
  }

  int n_inequality = dim_ineq_ + dim_var_;
  Eigen::MatrixXd C_with_bound(n_inequality, dim_var_);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var_, dim_var_);

  C_with_bound.topRows(dim_ineq_) = C;
  C_with_bound.bottomRows(dim_var_) = I;

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> C_sparse = C_with_bound.sparseView();
    proxqp_sparse_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, C_sparse, std::nullopt, std::nullopt);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, C_with_bound, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (d.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityVector] Dimension mismatch");
    return false;
  }

  constexpr double kBoundLimit = 1e8;
  Eigen::VectorXd d_with_bound_min(dim_ineq_ + dim_var_);
  Eigen::VectorXd d_with_bound_max(dim_ineq_ + dim_var_);
  d_with_bound_min.head(dim_ineq_) = Eigen::VectorXd::Constant(dim_ineq_, -1e8);
  d_with_bound_min.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, -kBoundLimit);
  d_with_bound_max.head(dim_ineq_) = d;
  d_with_bound_max.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, kBoundLimit);

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    proxqp_sparse_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, d_with_bound_min,
      d_with_bound_max);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, d_with_bound_min,
      d_with_bound_max);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateInequalityVectorBothSide(
  const Eigen::Ref<const Eigen::VectorXd> & d_lower,
  const Eigen::Ref<const Eigen::VectorXd> & d_upper)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateInequalityVectorBothSide] Solver not initialized. Call solve() "
      "first.");
    return false;
  }

  if (d_lower.size() != dim_ineq_ || d_upper.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateInequalityVectorBothSide] Dimension mismatch");
    return false;
  }

  constexpr double kBoundLimit = 1e8;
  Eigen::VectorXd l_with_bound(dim_ineq_ + dim_var_);
  Eigen::VectorXd u_with_bound(dim_ineq_ + dim_var_);
  l_with_bound.head(dim_ineq_) = d_lower;
  l_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, -kBoundLimit);
  u_with_bound.head(dim_ineq_) = d_upper;
  u_with_bound.tail(dim_var_) = Eigen::VectorXd::Constant(dim_var_, kBoundLimit);

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    proxqp_sparse_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, l_with_bound,
      u_with_bound);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, l_with_bound,
      u_with_bound);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateEqualityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (A.rows() != dim_eq_ || A.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateEqualityMatrix] Dimension mismatch");
    return false;
  }

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_sparse = A.sparseView();
    proxqp_sparse_->update(
      std::nullopt, std::nullopt, A_sparse, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, std::nullopt, A, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
    return true;
  }
  return false;
}
bool QpSolverProxqp::updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b)
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::updateEqualityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (b.size() != dim_eq_) {
    QSC_ERROR_STREAM("[QpSolverProxqp::updateEqualityVector] Dimension mismatch");
    return false;
  }

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    proxqp_sparse_->update(
      std::nullopt, std::nullopt, std::nullopt, b, std::nullopt, std::nullopt, std::nullopt);
    return true;
  } else if (proxqp_) {
    proxqp_->update(
      std::nullopt, std::nullopt, std::nullopt, b, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
    return true;
  }
  return false;
}
Eigen::VectorXd QpSolverProxqp::solveIncremental()
{
  if (!proxqp_ && !proxqp_sparse_) {
    QSC_ERROR_STREAM(
      "[QpSolverProxqp::solveIncremental] Solver not initialized. Call solve() first.");
    return Eigen::VectorXd::Zero(0);
  }

  auto solve_start_time = clock::now();

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    proxqp_sparse_->solve();
  } else if (proxqp_) {
    proxqp_->solve();
  }

  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
    if (proxqp_params_.use_sparse && proxqp_sparse_) {
      logger_->log("proxqp_iter", proxqp_sparse_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_sparse_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_sparse_->results.info.dua_res);
    } else if (proxqp_) {
      logger_->log("proxqp_iter", proxqp_->results.info.iter);
      logger_->log("proxqp_pri_res", proxqp_->results.info.pri_res);
      logger_->log("proxqp_dua_res", proxqp_->results.info.dua_res);
    }
  }

  if (proxqp_params_.use_sparse && proxqp_sparse_) {
    auto status = proxqp_sparse_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM(
        "[QpSolverProxqp::solveIncremental sparse] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_sparse_->results.x;
  } else if (proxqp_) {
    auto status = proxqp_->results.info.status;
    // Accept SOLVED, MAX_ITER_REACHED, and CLOSEST_PRIMAL_FEASIBLE as success
    if (
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE ||
      status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
      solve_failed_ = true;
      QSC_WARN_STREAM(
        "[QpSolverProxqp::solveIncremental] Failed to solve: " << static_cast<int>(status));
    } else {
      solve_failed_ = false;
    }
    return proxqp_->results.x;
  }
  return Eigen::VectorXd::Zero(0);
}
namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverProxqp() { return std::make_shared<QpSolverProxqp>(); }
}  // namespace QpSolverCollection
#endif
