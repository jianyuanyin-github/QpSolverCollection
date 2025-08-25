/* Author: Masaki Murooka */

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
  osqp_params_.max_iter = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.max_iter", 60, tam::pmg::ParameterType::INTEGER, "")
    .as_int();
  osqp_params_.abs_tolerance = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.abs_tolerance", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  osqp_params_.rel_tolerance = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.rel_tolerance", 1e-3, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  osqp_params_.alpha = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.alpha", 1.6, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  osqp_params_.verbose = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.verbose", false, tam::pmg::ParameterType::BOOL, "")
    .as_bool();
  osqp_params_.scaling = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.scaling", 10, tam::pmg::ParameterType::INTEGER, "")
    .as_int();
  osqp_params_.polish = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.polish", false, tam::pmg::ParameterType::BOOL, "")
    .as_bool();
  osqp_params_.check_termination = param_manager_
    ->declare_and_get_value("MPC.Solver_OSQP.check_termination", 0, tam::pmg::ParameterType::INTEGER, "")
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
  AC_with_bound << A, C, I;
  bd_with_bound_min << b,
    Eigen::VectorXd::Constant(dim_ineq, -1 * std::numeric_limits<double>::infinity()), x_min;
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

  auto status = osqp_->solveProblem();

  if (status == OsqpEigen::ErrorExitFlag::NoError) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solve] Failed to solve: " << to_string(status));
  }

  return osqp_->getSolution();
}
// ===== INCREMENTAL UPDATE IMPLEMENTATION =====
bool QpSolverOsqp::updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q)
{
  // Convert to sparse and update
  Q_sparse_ = Q.sparseView();
  osqp_->updateHessianMatrix(Q_sparse_);
  return true;
}
bool QpSolverOsqp::updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c)
{
  // Update gradient vector
  c_ = c;
  osqp_->updateGradient(c_);
  return true;
}
bool QpSolverOsqp::updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C)
{
  int dim_var = C.cols();
  int dim_ineq = C.rows();
  int dim_eq_ineq_with_bound = dim_ineq + dim_var;

  Eigen::MatrixXd AC_with_bound(dim_eq_ineq_with_bound, dim_var);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);
  AC_with_bound << C, I;
  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);
  return true;
}
bool QpSolverOsqp::updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d)
{
  if (!osqp_ || !osqp_->isInitialized()) return false;

  int dim_ineq = d.size();

  // Update only inequality bounds (no equality constraints)
  bd_with_bound_max_.head(dim_ineq) = d;

  osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  return true;
}
Eigen::VectorXd QpSolverOsqp::solveIncremental()
{
  auto status = osqp_->solveProblem();

  if (status == OsqpEigen::ErrorExitFlag::NoError) {
    solve_failed_ = false;
    return osqp_->getSolution();
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solveIncremental] Failed to solve: " << to_string(status));
    return Eigen::VectorXd::Zero(osqp_->data()->getData()->n);
  }
}
namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverOsqp() { return std::make_shared<QpSolverOsqp>(); }
}  // namespace QpSolverCollection
#endif
