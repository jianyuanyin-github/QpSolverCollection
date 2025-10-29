/* Author: Masaki Murooka */

#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_QPOASES
#include <qp_solver_collection/QpSolverCollection.h>

#include <limits>
#include <qpOASES.hpp>

using namespace QpSolverCollection;
QpSolverQpoases::QpSolverQpoases()
{
  type_ = QpSolverType::qpOASES;

  // Initialize qpOASES parameters
  declare_and_update_parameters();
}
void QpSolverQpoases::declare_and_update_parameters()
{
  qpoases_params_.max_iter =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.max_iter", 10, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  qpoases_params_.termination_tolerance =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.termination_tolerance", 1e-2, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  qpoases_params_.bound_tolerance =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.bound_tolerance", 1e-4, tam::pmg::ParameterType::DOUBLE, "")
      .as_double();
  qpoases_params_.enable_cholesky_refactorisation =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.enable_cholesky_refactorisation", false, tam::pmg::ParameterType::BOOL,
        "")
      .as_bool();
  qpoases_params_.enable_regularisation =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.enable_regularisation", true, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  std::cout << "enable_regularisation: " << qpoases_params_.enable_regularisation << std::endl;
  qpoases_params_.use_warm_start =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.use_warm_start", true, tam::pmg::ParameterType::BOOL, "")
      .as_bool();
  qpoases_params_.num_refinement_steps =
    param_manager_
      ->declare_and_get_value(
        "MPC.Solver_qpOASES.num_refinement_steps", 1, tam::pmg::ParameterType::INTEGER, "")
      .as_int();
  previous_param_state_hash_ = param_manager_->get_state_hash();
}
Eigen::VectorXd QpSolverQpoases::solve(
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

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AC_row_major(
    dim_eq + dim_ineq, dim_var);
  Eigen::VectorXd bd_min(dim_eq + dim_ineq);
  Eigen::VectorXd bd_max(dim_eq + dim_ineq);
  // Vertically stack A and C
  AC_row_major.topRows(dim_eq) = A;
  AC_row_major.bottomRows(dim_ineq) = C;
  bd_min << b, Eigen::VectorXd::Constant(dim_ineq, -1 * std::numeric_limits<double>::infinity());
  bd_max << b, d;
  n_wsr_ = qpoases_params_.max_iter;
  qpOASES::returnValue status = qpOASES::TERMINAL_LIST_ELEMENT;
  if (
    qpoases_params_.use_warm_start && !solve_failed_ && !force_initialize_ && qpoases_ &&
    qpoases_->getNV() == dim_var && qpoases_->getNC() == dim_eq + dim_ineq) {
    auto solve_start_time = clock::now();
    status = qpoases_->hotstart(
      // Since Q is a symmetric matrix, row/column-majors are interchangeable
      Q.data(), c.data(), AC_row_major.data(), x_min.data(), x_max.data(), bd_min.data(),
      bd_max.data(), n_wsr_);
    auto solve_end_time = clock::now();
    solve_time_us_ =
      std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
        .count();
  }
  if (status != qpOASES::SUCCESSFUL_RETURN) {
    qpoases_ = std::make_unique<qpOASES::SQProblem>(dim_var, dim_eq + dim_ineq);
    qpoases_->setPrintLevel(qpOASES::PL_LOW);

    qpOASES::Options options;
    options.setToMPC();
    options.terminationTolerance = qpoases_params_.termination_tolerance;
    options.boundTolerance = qpoases_params_.bound_tolerance;
    options.enableCholeskyRefactorisation =
      qpoases_params_.enable_cholesky_refactorisation ? qpOASES::BT_TRUE : qpOASES::BT_FALSE;
    options.enableRegularisation =
      qpoases_params_.enable_regularisation ? qpOASES::BT_TRUE : qpOASES::BT_FALSE;
    options.numRefinementSteps = qpoases_params_.num_refinement_steps;

    qpoases_->setOptions(options);

    auto solve_start_time = clock::now();
    status = qpoases_->init(
      // Since Q is a symmetric matrix, row/column-majors are interchangeable
      Q.data(), c.data(), AC_row_major.data(), x_min.data(), x_max.data(), bd_min.data(),
      bd_max.data(), n_wsr_);
    auto solve_end_time = clock::now();
    solve_time_us_ =
      std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
        .count();
  }

  if (status == qpOASES::SUCCESSFUL_RETURN) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverQpoases::solve] Failed to solve: " << static_cast<int>(status));
  }

  // Log solver time for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
  }

  Eigen::VectorXd sol(dim_var);
  qpoases_->getPrimalSolution(sol.data());
  return sol;
}
Eigen::VectorXd QpSolverQpoases::solve(
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

  // qpOASES supports bilateral constraints natively
  int dim_constraints = dim_eq + dim_ineq;
  Eigen::MatrixXd AC(dim_constraints, dim_var);
  Eigen::VectorXd bd_min(dim_constraints);
  Eigen::VectorXd bd_max(dim_constraints);

  // Vertically stack A and C
  AC.topRows(dim_eq) = A;
  AC.bottomRows(dim_ineq) = C;
  // Use bilateral constraints: d_lower <= Cx <= d_upper
  bd_min << b, d_lower;
  bd_max << b, d_upper;

  n_wsr_ = qpoases_params_.max_iter;
  qpOASES::returnValue status = qpOASES::TERMINAL_LIST_ELEMENT;
  if (
    qpoases_params_.use_warm_start && !solve_failed_ && !force_initialize_ && qpoases_ &&
    qpoases_->getNV() == dim_var && qpoases_->getNC() == dim_constraints) {
    // Hot start
    auto solve_start_time = clock::now();
    status = qpoases_->hotstart(
      Q.data(), c.data(), AC.data(), x_min.data(), x_max.data(), bd_min.data(), bd_max.data(),
      n_wsr_);
    auto solve_end_time = clock::now();
    solve_time_us_ =
      std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
        .count();
  } else {
    // Cold start
    qpoases_ = std::make_unique<qpOASES::SQProblem>(dim_var, dim_constraints);
    qpoases_->setPrintLevel(qpOASES::PL_LOW);

    // Set options
    qpOASES::Options options;
    options.setToMPC();
    options.terminationTolerance = qpoases_params_.termination_tolerance;
    options.boundTolerance = qpoases_params_.bound_tolerance;
    options.enableCholeskyRefactorisation =
      qpoases_params_.enable_cholesky_refactorisation ? qpOASES::BT_TRUE : qpOASES::BT_FALSE;
    options.enableRegularisation =
      qpoases_params_.enable_regularisation ? qpOASES::BT_TRUE : qpOASES::BT_FALSE;
    options.numRefinementSteps = qpoases_params_.num_refinement_steps;
    qpoases_->setOptions(options);

    auto solve_start_time = clock::now();
    status = qpoases_->init(
      Q.data(), c.data(), AC.data(), x_min.data(), x_max.data(), bd_min.data(), bd_max.data(),
      n_wsr_);
    auto solve_end_time = clock::now();
    solve_time_us_ =
      std::chrono::duration_cast<std::chrono::microseconds>(solve_end_time - solve_start_time)
        .count();
  }

  if (status == qpOASES::SUCCESSFUL_RETURN) {
    solve_failed_ = false;
  } else {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverQpoases::solve bilateral] Failed to solve: " << status);
  }

  // Log solver time for performance analysis
  if (logger_) {
    logger_->log("solver_time_pure", solve_time_us_);
  }

  Eigen::VectorXd sol(dim_var);
  qpoases_->getPrimalSolution(sol.data());
  return sol;
}
namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverQpoases() { return std::make_shared<QpSolverQpoases>(); }
}  // namespace QpSolverCollection
#endif
