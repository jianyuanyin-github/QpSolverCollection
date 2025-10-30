
#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_OSQP
#include <OsqpEigen/OsqpEigen.h>
#include <qp_solver_collection/QpSolverCollection.h>

#include <limits>
#define OSQP_EIGEN_DEBUG_OUTPUT
static inline std::string to_string(OsqpEigen::ErrorExitFlag flag)
{
  switch (flag) {
    case OsqpEigen::ErrorExitFlag::NoError:
      return "No error";
    case OsqpEigen::ErrorExitFlag::DataValidationError:
      return "Data validation error";
    case OsqpEigen::ErrorExitFlag::SettingsValidationError:
      return "Settings validation error";
    case OsqpEigen::ErrorExitFlag::LinsysSolverInitError:
      return "Linsys solver initialization error";
    case OsqpEigen::ErrorExitFlag::NonCvxError:
      return "Non convex error";
    case OsqpEigen::ErrorExitFlag::MemAllocError:
      return "Mem alloc error";
    case OsqpEigen::ErrorExitFlag::WorkspaceNotInitError:
      return "Workspace not initialized error";
    default:
      return "Unknown value: " +
             std::to_string(static_cast<std::underlying_type_t<OsqpEigen::ErrorExitFlag>>(flag));
  }
}
using namespace QpSolverCollection;
QpSolverOsqp::QpSolverOsqp()
{
  type_ = QpSolverType::OSQP;
  osqp_ = std::make_unique<OsqpEigen::Solver>();

  // Initialize OSQP parameters
  declare_and_update_parameters();
}
void QpSolverOsqp::declare_and_update_parameters()
{
  osqp_params_.max_iter =
    param_manager_
      ->declare_and_get_value("MPC.Solver_OSQP.max_iter", 60, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  osqp_params_.abs_tolerance =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_OSQP.abs_tolerance", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  osqp_params_.rel_tolerance =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_OSQP.rel_tolerance", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  osqp_params_.alpha =
    param_manager_
      ->declare_and_get_value("MPC.Solver_OSQP.alpha", 1.6, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  osqp_params_.verbose =
    param_manager_
      ->declare_and_get_value("MPC.Solver_OSQP.verbose", false, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  osqp_params_.scaling =
    param_manager_
      ->declare_and_get_value("MPC.Solver_OSQP.scaling", 10, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  osqp_params_.polish =
    param_manager_
      ->declare_and_get_value("MPC.Solver_OSQP.polish", false, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  osqp_params_.check_termination =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_OSQP.check_termination", 0, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  previous_param_state_hash_ = param_manager_->get_state_hash();
}
Eigen::VectorXd QpSolverOsqp::solve(
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

  int dim_eq_ineq_with_bound = dim_eq + dim_ineq + dim_var;
  Eigen::MatrixXd AC_with_bound(dim_eq_ineq_with_bound, dim_var);
  Eigen::VectorXd bd_with_bound_min(dim_eq_ineq_with_bound);
  Eigen::VectorXd bd_with_bound_max(dim_eq_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);

  // Correctly stack matrices row-wise (vertical), NOT column-wise
  AC_with_bound.topRows(dim_eq) = A;
  AC_with_bound.middleRows(dim_eq, dim_ineq) = C;
  AC_with_bound.bottomRows(dim_var) = I;

  bd_with_bound_min << b, Eigen::VectorXd::Constant(dim_ineq, -1e30), x_min;
  bd_with_bound_max << b, d, x_max;

  auto sparse_start_time = clock::now();
  // Matrices and vectors must be hold during solver's lifetime
  Q_sparse_ = Q.sparseView();
  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  // You must pass unconst vectors to OSQP
  c_ = c;
  bd_with_bound_min_ = bd_with_bound_min;
  bd_with_bound_max_ = bd_with_bound_max;
  auto sparse_end_time = clock::now();
  sparse_duration_ = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
                             sparse_end_time - sparse_start_time)
                             .count();

  // Apply all OSQP settings from parameters
  osqp_->settings()->setMaxIteration(osqp_params_.max_iter);
  osqp_->settings()->setAbsoluteTolerance(osqp_params_.abs_tolerance);
  osqp_->settings()->setRelativeTolerance(osqp_params_.rel_tolerance);
  osqp_->settings()->setAlpha(osqp_params_.alpha);
  osqp_->settings()->setVerbosity(osqp_params_.verbose);
  osqp_->settings()->setScaling(osqp_params_.scaling);
  // osqp_->settings()->setAdaptiveRhoInterval(osqp_params_.adaptive_rho_interval);
  osqp_->settings()->setPolish(osqp_params_.polish);
  // osqp_->settings()->setTimeLimit(osqp_params_.time_limit);
  osqp_->settings()->setCheckTermination(osqp_params_.check_termination);
  osqp_->settings()->setWarmStart(true);

  // Store dimensions for incremental updates
  dim_var_ = dim_var;
  dim_eq_ = dim_eq;
  dim_ineq_ = dim_ineq;

  if (
    !solve_failed_ && !force_initialize_ && osqp_->isInitialized() &&
    dim_var == osqp_->data()->getData()->n &&
    dim_eq_ineq_with_bound == osqp_->data()->getData()->m) {
    // Update only matrices and vectors
    osqp_->updateHessianMatrix(Q_sparse_);
    osqp_->updateGradient(c_);
    osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  } else {
    // Initialize fully
    if (osqp_->isInitialized()) {
      osqp_->clearSolver();
      osqp_->data()->clearHessianMatrix();
      osqp_->data()->clearLinearConstraintsMatrix();
    }

    osqp_->data()->setNumberOfVariables(dim_var);
    osqp_->data()->setNumberOfConstraints(dim_eq_ineq_with_bound);
    osqp_->data()->setHessianMatrix(Q_sparse_);
    osqp_->data()->setGradient(c_);
    osqp_->data()->setLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->data()->setLowerBound(bd_with_bound_min_);
    osqp_->data()->setUpperBound(bd_with_bound_max_);
    osqp_->initSolver();
  }

  auto solve_start_time = clock::now();
  auto status = osqp_->solveProblem();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Get solver status and diagnostics
  auto solver_status = osqp_->getStatus();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
  }

  if (status == OsqpEigen::ErrorExitFlag::NoError) {
    solve_failed_ = false;
    if (solver_status == OsqpEigen::Status::SolvedInaccurate) {
      QSC_WARN_STREAM("[QpSolverOsqp::solve] Solved with reduced accuracy");
    } else if (solver_status == OsqpEigen::Status::MaxIterReached) {
      QSC_WARN_STREAM("[QpSolverOsqp::solve] MAX_ITER reached without convergence");
    }
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solve] Failed to solve: " << to_string(status));
  }

  return osqp_->getSolution();
}
Eigen::VectorXd QpSolverOsqp::solve(
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

  int dim_eq_ineq_with_bound = dim_eq + dim_ineq + dim_var;
  Eigen::MatrixXd AC_with_bound(dim_eq_ineq_with_bound, dim_var);
  Eigen::VectorXd bd_with_bound_min(dim_eq_ineq_with_bound);
  Eigen::VectorXd bd_with_bound_max(dim_eq_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);

  // Correctly stack matrices row-wise (vertical), NOT column-wise
  AC_with_bound.topRows(dim_eq) = A;
  AC_with_bound.middleRows(dim_eq, dim_ineq) = C;
  AC_with_bound.bottomRows(dim_var) = I;

  // Use bilateral constraints: d_lower <= Cx <= d_upper
  bd_with_bound_min << b, d_lower, x_min;
  bd_with_bound_max << b, d_upper, x_max;

  auto sparse_start_time = clock::now();
  // Matrices and vectors must be hold during solver's lifetime
  Q_sparse_ = Q.sparseView();
  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  // You must pass unconst vectors to OSQP
  c_ = c;
  bd_with_bound_min_ = bd_with_bound_min;
  bd_with_bound_max_ = bd_with_bound_max;
  auto sparse_end_time = clock::now();
  sparse_duration_ = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
                             sparse_end_time - sparse_start_time)
                             .count();

  // Apply all OSQP settings from parameters
  osqp_->settings()->setMaxIteration(osqp_params_.max_iter);
  osqp_->settings()->setAbsoluteTolerance(osqp_params_.abs_tolerance);
  osqp_->settings()->setRelativeTolerance(osqp_params_.rel_tolerance);
  osqp_->settings()->setAlpha(osqp_params_.alpha);
  osqp_->settings()->setVerbosity(osqp_params_.verbose);
  osqp_->settings()->setScaling(osqp_params_.scaling);
  osqp_->settings()->setPolish(osqp_params_.polish);
  osqp_->settings()->setCheckTermination(osqp_params_.check_termination);
  osqp_->settings()->setWarmStart(true);

  // Store dimensions for incremental updates
  dim_var_ = dim_var;
  dim_eq_ = dim_eq;
  dim_ineq_ = dim_ineq;

  if (
    !solve_failed_ && !force_initialize_ && osqp_->isInitialized() &&
    dim_var == osqp_->data()->getData()->n &&
    dim_eq_ineq_with_bound == osqp_->data()->getData()->m) {
    // Update only matrices and vectors
    osqp_->updateHessianMatrix(Q_sparse_);
    osqp_->updateGradient(c_);
    osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  } else {
    // Initialize fully
    if (osqp_->isInitialized()) {
      osqp_->clearSolver();
      osqp_->data()->clearHessianMatrix();
      osqp_->data()->clearLinearConstraintsMatrix();
    }

    osqp_->data()->setNumberOfVariables(dim_var);
    osqp_->data()->setNumberOfConstraints(dim_eq_ineq_with_bound);
    osqp_->data()->setHessianMatrix(Q_sparse_);
    osqp_->data()->setGradient(c_);
    osqp_->data()->setLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->data()->setLowerBound(bd_with_bound_min_);
    osqp_->data()->setUpperBound(bd_with_bound_max_);
    osqp_->initSolver();
  }

  auto solve_start_time = clock::now();
  auto status = osqp_->solveProblem();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  // Get solver status and diagnostics
  auto solver_status = osqp_->getStatus();

  // Log solver time and diagnostics for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
  }

  // Check solver status
  if (status == OsqpEigen::ErrorExitFlag::NoError) {
    solve_failed_ = false;
    if (solver_status == OsqpEigen::Status::SolvedInaccurate) {
      QSC_WARN_STREAM("[QpSolverOsqp::solve bilateral] Solved with reduced accuracy");
    } else if (solver_status == OsqpEigen::Status::MaxIterReached) {
      QSC_WARN_STREAM("[QpSolverOsqp::solve bilateral] MAX_ITER reached without convergence");
    }
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solve bilateral] Failed to solve: " << to_string(status));
  }

  return osqp_->getSolution();
}
// ===== INCREMENTAL UPDATE IMPLEMENTATION =====
bool QpSolverOsqp::updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateObjectiveMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (Q.rows() != dim_var_ || Q.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateObjectiveMatrix] Dimension mismatch");
    return false;
  }

  Q_sparse_ = Q.sparseView();
  osqp_->updateHessianMatrix(Q_sparse_);
  return true;
}
bool QpSolverOsqp::updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateObjectiveVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (c.size() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateObjectiveVector] Dimension mismatch");
    return false;
  }

  c_ = c;
  osqp_->updateGradient(c_);
  return true;
}
bool QpSolverOsqp::updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateInequalityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (C.rows() != dim_ineq_ || C.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateInequalityMatrix] Dimension mismatch");
    return false;
  }

  // Rebuild full constraint matrix: [A; C; I]
  int total_constraints = dim_eq_ + dim_ineq_ + dim_var_;
  Eigen::MatrixXd AC_with_bound(total_constraints, dim_var_);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var_, dim_var_);

  // Extract current equality constraints from existing matrix (if available)
  if (dim_eq_ > 0) {
    // Convert sparse to dense properly
    Eigen::MatrixXd AC_current(AC_with_bound_sparse_);
    AC_with_bound.topRows(dim_eq_) = AC_current.topRows(dim_eq_);
  }

  // Set new inequality constraints
  AC_with_bound.middleRows(dim_eq_, dim_ineq_) = C;

  // Set variable bounds (identity matrix)
  AC_with_bound.bottomRows(dim_var_) = I;

  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);
  return true;
}
bool QpSolverOsqp::updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateInequalityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (d.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateInequalityVector] Dimension mismatch");
    return false;
  }

  // Update inequality bounds using correct offset: [b; d; x_max]
  // Inequality constraints start at offset dim_eq_
  bd_with_bound_max_.segment(dim_eq_, dim_ineq_) = d;

  // Lower bounds for inequality constraints remain -1e30
  bd_with_bound_min_.segment(dim_eq_, dim_ineq_) = Eigen::VectorXd::Constant(dim_ineq_, -1e30);

  osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  return true;
}
bool QpSolverOsqp::updateInequalityVectorBothSide(
  const Eigen::Ref<const Eigen::VectorXd> & d_lower,
  const Eigen::Ref<const Eigen::VectorXd> & d_upper)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateInequalityVectorBothSide] Solver not initialized. Call solve() first.");
    return false;
  }

  if (d_lower.size() != dim_ineq_ || d_upper.size() != dim_ineq_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateInequalityVectorBothSide] Dimension mismatch");
    return false;
  }

  // Update inequality bounds using correct offset: [b; d_lower/d_upper; x_min/x_max]
  // Inequality constraints start at offset dim_eq_
  bd_with_bound_min_.segment(dim_eq_, dim_ineq_) = d_lower;
  bd_with_bound_max_.segment(dim_eq_, dim_ineq_) = d_upper;

  osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  return true;
}
bool QpSolverOsqp::updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateEqualityMatrix] Solver not initialized. Call solve() first.");
    return false;
  }

  if (A.rows() != dim_eq_ || A.cols() != dim_var_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateEqualityMatrix] Dimension mismatch");
    return false;
  }

  // Rebuild full constraint matrix: [A; C; I]
  int total_constraints = dim_eq_ + dim_ineq_ + dim_var_;
  Eigen::MatrixXd AC_with_bound(total_constraints, dim_var_);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var_, dim_var_);

  // Set new equality constraints at the top
  AC_with_bound.topRows(dim_eq_) = A;

  // Extract existing inequality constraints and variable bounds
  if (dim_ineq_ > 0 || dim_var_ > 0) {
    // Convert sparse to dense properly
    Eigen::MatrixXd AC_current(AC_with_bound_sparse_);
    // Copy inequality constraints
    if (dim_ineq_ > 0) {
      AC_with_bound.middleRows(dim_eq_, dim_ineq_) = AC_current.middleRows(dim_eq_, dim_ineq_);
    }
    // Set variable bounds (identity matrix)
    AC_with_bound.bottomRows(dim_var_) = I;
  }

  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);

  return true;
}
bool QpSolverOsqp::updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b)
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::updateEqualityVector] Solver not initialized. Call solve() first.");
    return false;
  }

  if (b.size() != dim_eq_) {
    QSC_ERROR_STREAM("[QpSolverOsqp::updateEqualityVector] Dimension mismatch");
    return false;
  }

  // Update equality constraint bounds: [b; d; x_min/x_max]
  // Equality constraints are at the beginning (offset 0)
  bd_with_bound_min_.head(dim_eq_) = b;
  bd_with_bound_max_.head(dim_eq_) = b;

  osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);

  return true;
}
Eigen::VectorXd QpSolverOsqp::solveIncremental()
{
  if (!osqp_ || !osqp_->isInitialized()) {
    QSC_ERROR_STREAM(
      "[QpSolverOsqp::solveIncremental] Solver not initialized. Call solve() first.");
    return Eigen::VectorXd::Zero(0);
  }

  auto solve_start_time = clock::now();
  auto status = osqp_->solveProblem();
  auto solve_end_time = clock::now();
  solve_time_us_ =
    std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
      .count();

  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
  }

  auto solver_status = osqp_->getStatus();
  if (status == OsqpEigen::ErrorExitFlag::NoError) {
    solve_failed_ = false;
    if (solver_status == OsqpEigen::Status::SolvedInaccurate) {
      QSC_WARN_STREAM("[QpSolverOsqp::solveIncremental] Solved with reduced accuracy");
    } else if (solver_status == OsqpEigen::Status::MaxIterReached) {
      QSC_WARN_STREAM("[QpSolverOsqp::solveIncremental] MAX_ITER reached without convergence");
    }
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solveIncremental] Failed to solve: " << to_string(status));
  }

  return osqp_->getSolution();
}
namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverOsqp() { return std::make_shared<QpSolverOsqp>(); }
}  // namespace QpSolverCollection
#endif
