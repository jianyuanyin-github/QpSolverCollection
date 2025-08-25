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
        "MPC.Solver_PROXQP.eps_abs", 1e-6, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  proxqp_params_.eps_rel =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.eps_rel", 1e-6, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  proxqp_params_.max_iter =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_PROXQP.max_iter", 50, tam::pmg::ParameterType::INTEGER, "")
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

  int dim_ineq_with_bound = dim_ineq + dim_var;
  if (!(proxqp_ && proxqp_->model.dim == dim_var && proxqp_->model.n_eq == dim_eq &&
        proxqp_->model.n_in == dim_ineq_with_bound)) {
    proxqp_ =
      std::make_unique<proxsuite::proxqp::dense::QP<double>>(dim_var, dim_eq, dim_ineq_with_bound);

    proxqp_->settings.eps_abs = proxqp_params_.eps_abs;
    proxqp_->settings.eps_rel = proxqp_params_.eps_rel;
    proxqp_->settings.max_iter = proxqp_params_.max_iter;
    proxqp_->settings.verbose = proxqp_params_.verbose;
    proxqp_->settings.compute_timings = proxqp_params_.compute_timings;
    proxqp_->settings.check_duality_gap = proxqp_params_.check_duality_gap;
  }

  Eigen::MatrixXd C_with_bound(dim_ineq_with_bound, dim_var);
  Eigen::VectorXd d_with_bound_min(dim_ineq_with_bound);
  Eigen::VectorXd d_with_bound_max(dim_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);
  C_with_bound << C, I;
  d_with_bound_min << Eigen::VectorXd::Constant(
    dim_ineq, -1 * std::numeric_limits<double>::infinity()),
    x_min;
  d_with_bound_max << d, x_max;

  proxqp_->update(Q, c, A, b, C_with_bound, d_with_bound_min, d_with_bound_max);
  proxqp_->solve();

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
// ===== INCREMENTAL UPDATE IMPLEMENTATION =====
bool QpSolverProxqp::updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q)
{
  proxqp_->update(
    Q, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c)
{
  proxqp_->update(
    std::nullopt, c, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C)
{
  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, std::nullopt, C, std::nullopt, std::nullopt,
    std::nullopt, std::nullopt);
  return true;
}
bool QpSolverProxqp::updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d)
{
  // ProxQP uses upper bounds, so we need to update both lower and upper
  proxqp_->update(
    std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, d,
    std::nullopt, std::nullopt);
  return true;
}
Eigen::VectorXd QpSolverProxqp::solveIncremental()
{
  proxqp_->solve();

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
