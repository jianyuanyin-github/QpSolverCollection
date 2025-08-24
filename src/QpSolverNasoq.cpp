/* Author: Masaki Murooka */

#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_NASOQ
#  include <qp_solver_collection/QpSolverCollection.h>

#  include <nasoq/nasoq_eigen.h>

using namespace QpSolverCollection;

QpSolverNasoq::QpSolverNasoq()
{
  type_ = QpSolverType::NASOQ;
  
  // Initialize NASOQ parameters
  declare_and_update_parameters();
}

void QpSolverNasoq::declare_and_update_parameters()
{
  nasoq_params_.max_iter = param_manager_
    ->declare_and_get_value("MPC.Solver_NASOQ.max_iter", 10, tam::pmg::ParameterType::INTEGER, "")
    .as_int();
  nasoq_params_.eps_abs = param_manager_
    ->declare_and_get_value("MPC.Solver_NASOQ.eps_abs", 1e-5, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  nasoq_params_.eps_rel = param_manager_
    ->declare_and_get_value("MPC.Solver_NASOQ.eps_rel", 1e-5, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  nasoq_params_.regularization = param_manager_
    ->declare_and_get_value("MPC.Solver_NASOQ.regularization", 1e-9, tam::pmg::ParameterType::DOUBLE, "")
    .as_double();
  nasoq_params_.nasoq_variant = param_manager_
    ->declare_and_get_value("MPC.Solver_NASOQ.nasoq_variant", std::string("auto"), tam::pmg::ParameterType::STRING, "")
    .as_string();
  previous_param_state_hash_ = param_manager_->get_state_hash();
}

Eigen::VectorXd QpSolverNasoq::solve(int dim_var,
                                     int dim_eq,
                                     int dim_ineq,
                                     Eigen::Ref<Eigen::MatrixXd> Q,
                                     const Eigen::Ref<const Eigen::VectorXd> & c,
                                     const Eigen::Ref<const Eigen::MatrixXd> & A,
                                     const Eigen::Ref<const Eigen::VectorXd> & b,
                                     const Eigen::Ref<const Eigen::MatrixXd> & C,
                                     const Eigen::Ref<const Eigen::VectorXd> & d,
                                     const Eigen::Ref<const Eigen::VectorXd> & x_min,
                                     const Eigen::Ref<const Eigen::VectorXd> & x_max)
{
  // Check if parameters have changed and update if necessary
  if (param_manager_->get_state_hash() != previous_param_state_hash_) {
    declare_and_update_parameters();
  }
  
  int dim_ineq_with_bound = dim_ineq + 2 * dim_var;
  Eigen::MatrixXd C_with_bound(dim_ineq_with_bound, dim_var);
  Eigen::VectorXd d_with_bound(dim_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);
  C_with_bound << C, I, -I;
  d_with_bound << d, x_max, -x_min;

  auto sparse_start_time = clock::now();
  // Matrices and vectors must be hold during solver's lifetime
  Q_sparse_ = Q.sparseView();
  A_sparse_ = A.sparseView();
  C_with_bound_sparse_ = C_with_bound.sparseView();
  auto sparse_end_time = clock::now();
  sparse_duration_ =
      1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(sparse_end_time - sparse_start_time).count();

  Eigen::VectorXd sol(dim_var), dual_eq(dim_eq), dual_ineq(dim_ineq_with_bound);
  nasoq::QPSettings settings;
  settings.eps = nasoq_params_.eps_abs;
  settings.eps_rel = nasoq_params_.eps_rel;
  settings.diag_perturb = nasoq_params_.regularization;
  settings.max_iter = nasoq_params_.max_iter;
  settings.nasoq_variant = nasoq_params_.nasoq_variant;

  int solve_ret = nasoq::quadprog(Q_sparse_.triangularView<Eigen::Lower>(), c, A_sparse_, b, C_with_bound_sparse_,
                                  d_with_bound, sol, dual_eq, dual_ineq, &settings);

  if(solve_ret == nasoq::Optimal)
  {
    solve_failed_ = false;
  }
  else
  {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverNasoq::solve] Failed to solve: " << solve_ret);
  }

  return sol;
}

namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverNasoq()
{
  return std::make_shared<QpSolverNasoq>();
}
} // namespace QpSolverCollection
#endif
