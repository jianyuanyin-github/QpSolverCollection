/* Author: Masaki Murooka */

#pragma once

#include <qp_solver_collection/QpSolverOptions.h>

#include <Eigen/SparseCore>
#include <chrono>
#include <fstream>
#include <memory>

#ifdef QP_SOLVER_COLLECTION_STANDALONE
#include <iostream>
#define QSC_ERROR_STREAM(x) std::cerr << x << "\n"
#define QSC_WARN_STREAM(x) std::cerr << x << "\n"
#define QSC_INFO_STREAM(x) std::cout << x << "\n"
#else
#include <rclcpp/rclcpp.hpp>

#include "param_management_cpp/param_value_manager.hpp"
#include "tsl_logger_cpp/value_logger.hpp"
#define QSC_ERROR_STREAM(msg) RCLCPP_ERROR_STREAM(rclcpp::get_logger("QpSolverCollection"), msg)
#define QSC_WARN_STREAM(msg) RCLCPP_WARN_STREAM(rclcpp::get_logger("QpSolverCollection"), msg)
#define QSC_INFO_STREAM(msg) RCLCPP_INFO_STREAM(rclcpp::get_logger("QpSolverCollection"), msg)
#endif
namespace Eigen
{
class QLDDirect;
class QuadProgDense;
class LSSOL_QP;
}  // namespace Eigen
namespace jrl
{
namespace qp
{
class GoldfarbIdnaniSolver;
}  // namespace qp
}  // namespace jrl
namespace qpOASES
{
class SQProblem;
}  // namespace qpOASES
namespace OsqpEigen
{
class Solver;
}  // namespace OsqpEigen

struct d_dense_qp_dim;
struct d_dense_qp;
struct d_dense_qp_sol;
struct d_dense_qp_ipm_arg;
struct d_dense_qp_ipm_ws;
namespace proxsuite
{
namespace proxqp
{
namespace dense
{
template <typename T>
class QP;
}
}  // namespace proxqp
}  // namespace proxsuite
namespace qpmad
{
template <typename t_Scalar, int... t_Parameters>
class SolverTemplate;
using Solver = SolverTemplate<double, Eigen::Dynamic, 1, Eigen::Dynamic>;
}  // namespace qpmad
namespace QpSolverCollection
{
/** \brief QP solver type. */
enum class QpSolverType {
  Any = -2,
  Uninitialized = -1,
  QLD = 0,
  QuadProg,
  LSSOL,
  JRLQP,
  qpOASES,
  OSQP,
  NASOQ,
  HPIPM,
  PROXQP,
  QPMAD
};

/*! \brief Convert std::string to QpSolverType. */
QpSolverType strToQpSolverType(const std::string & qp_solver_type);
}  // namespace QpSolverCollection
namespace std
{
using QpSolverType = QpSolverCollection::QpSolverType;
inline string to_string(QpSolverType qp_solver_type)
{
  switch (qp_solver_type) {
    case QpSolverType::QLD:
      return "QpSolverType::QLD";
    case QpSolverType::QuadProg:
      return "QpSolverType::QuadProg";
    case QpSolverType::LSSOL:
      return "QpSolverType::LSSOL";
    case QpSolverType::JRLQP:
      return "QpSolverType::JRLQP";
    case QpSolverType::qpOASES:
      return "QpSolverType::qpOASES";
    case QpSolverType::OSQP:
      return "QpSolverType::OSQP";
    case QpSolverType::NASOQ:
      return "QpSolverType::NASOQ";
    case QpSolverType::HPIPM:
      return "QpSolverType::HPIPM";
    case QpSolverType::PROXQP:
      return "QpSolverType::PROXQP";
    case QpSolverType::QPMAD:
      return "QpSolverType::QPMAD";
    default:
      QSC_ERROR_STREAM(
        "[QpSolverType] Unsupported value: " << std::to_string(static_cast<int>(qp_solver_type)));
  }

  return "";
}
}  // namespace std
namespace QpSolverCollection
{
/** \brief Class of QP coefficient. */
class QpCoeff
{
public:
  /** \brief Constructor. */
  QpCoeff() {}
  /** \brief Setup the coefficients with filling zero.
      \param dim_var dimension of decision variable
      \param dim_eq dimension of equality constraint
      \param dim_ineq dimension of inequality constraint
  */
  void setup(int dim_var, int dim_eq, int dim_ineq);

  /** \brief Print information. */
  void printInfo(bool verbose = false, const std::string & header = "") const;

  /** \brief Dump coefficients. */
  void dump(std::ofstream & ofs) const;

public:
  //! Dimension of decision variable
  int dim_var_ = 0;

  //! Dimension of equality constraint
  int dim_eq_ = 0;

  //! Dimension of inequality constraint
  int dim_ineq_ = 0;

  //! Objective matrix (corresponding to \f$\boldsymbol{Q}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::MatrixXd obj_mat_;

  //! Objective vector (corresponding to \f$\boldsymbol{c}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd obj_vec_;

  //! Equality constraint matrix (corresponding to \f$\boldsymbol{A}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::MatrixXd eq_mat_;

  //! Equality constraint vector (corresponding to \f$\boldsymbol{b}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd eq_vec_;

  //! Inequality constraint matrix (corresponding to \f$\boldsymbol{C}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::MatrixXd ineq_mat_;

  //! Inequality constraint vector (corresponding to \f$\boldsymbol{d}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd ineq_vec_;

  //! Equality constraint vector (corresponding to \f$\boldsymbol{b}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd ineq_vec_lb_;

  //! Equality constraint vector (corresponding to \f$\boldsymbol{b}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd ineq_vec_ub_;

  //! Lower bound (corresponding to \f$\boldsymbol{x}_{min}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd x_min_;

  //! Upper bound (corresponding to \f$\boldsymbol{x}_{max}\f$ in @ref QpSolver#solve
  //! "QpSolver::solve".)
  Eigen::VectorXd x_max_;
};
/** \brief Virtual class of QP solver. */
class QpSolver
{
public:
  using clock = typename std::conditional<
    std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock,
    std::chrono::steady_clock>::type;

public:
  /** \brief Constructor. */
  QpSolver() {}
  /** \brief Print information. */
  void printInfo(bool verbose = false, const std::string & header = "") const;

  /** \brief Solve QP.
      \param dim_var dimension of decision variable
      \param dim_eq dimension of equality constraint
      \param dim_ineq dimension of inequality constraint
      \param Q objective matrix (LSSOL requires non-const for Q)
      \param c objective vector
      \param A equality constraint matrix
      \param b equality constraint vector
      \param C inequality constraint matrix
      \param d inequality constraint vector
      \param x_min lower bound
      \param x_max upper bound

      QP is formulated as follows:
      \f{align*}{
      & min_{\boldsymbol{x}} \ \frac{1}{2}{\boldsymbol{x}^T \boldsymbol{Q} \boldsymbol{x}} +
     {\boldsymbol{c}^T
     \boldsymbol{x}} \\
      & s.t. \ \ \boldsymbol{A} \boldsymbol{x} = \boldsymbol{b} \nonumber \\
      & \phantom{s.t.} \ \ \boldsymbol{C} \boldsymbol{x} \leq \boldsymbol{d} \nonumber \\
      & \phantom{s.t.} \ \ \boldsymbol{x}_{min} \leq \boldsymbol{x} \leq \boldsymbol{x}_{max}
     \nonumber \f}

      \todo Support both-sided inequality constraints (i.e., \f$\boldsymbol{d}_{lower} \leq
     \boldsymbol{C} \boldsymbol{x} \leq \boldsymbol{d}_{upper}\f$). QLD, QuadProg, and NASOQ support
     only one-sided constraints, while LSSOL, JRLQP, QPOASES, OSQP, HPIPM, PROXQP, and QPMAD support
     both-sided constraints.
  */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) = 0;

  /** \brief Solve QP.
      \param qp_coeff QP coefficient
  */
  virtual Eigen::VectorXd solve(QpCoeff & qp_coeff);
  /** \brief Solve QP with both-sided inequality constraints.
      \param dim_var dimension of decision variable
      \param dim_eq dimension of equality constraint
      \param dim_ineq dimension of inequality constraint
      \param Q objective matrix
      \param c objective vector
      \param A equality constraint matrix
      \param b equality constraint vector
      \param C inequality constraint matrix
      \param d_lower lower bounds for inequality constraints
      \param d_upper upper bounds for inequality constraints
      \param x_min lower bound
      \param x_max upper bound
  */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper,
    const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max)
  {
    QSC_ERROR_STREAM("[QpSolver] Bilateral constraints not supported by this solver");
    return Eigen::VectorXd::Zero(dim_var);
  }
  /** \brief Solve QP with both-sided inequality constraints using QpCoeff structure.
      \param qp_coeff QP coefficient containing bilateral constraint bounds
  */
  virtual Eigen::VectorXd solve_bothside(QpCoeff & qp_coeff);
  // ===== INCREMENTAL UPDATE INTERFACE =====
  /** \brief Check if solver supports incremental updates. */
  virtual bool supportsIncrementalUpdate() const { return false; }
  /** \brief Update objective matrix incrementally.
      \param Q new objective matrix
      \return true if update successful
  */
  virtual bool updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q) { return false; }
  /** \brief Update objective vector incrementally.
      \param c new objective vector
      \return true if update successful
  */
  virtual bool updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c) { return false; }
  /** \brief Update inequality constraint matrix incrementally.
      \param C new inequality constraint matrix
      \return true if update successful
  */
  virtual bool updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C) { return false; }
  /** \brief Update inequality constraint vector incrementally.
      \param d new inequality constraint vector
      \return true if update successful
  */
  virtual bool updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d) { return false; }
  /** \brief Update inequality constraint bounds incrementally (bilateral constraints).
      \param d_lower new lower bounds for inequality constraints
      \param d_upper new upper bounds for inequality constraints
      \return true if update successful
  */
  virtual bool updateInequalityVectorBothSide(
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper)
  {
    return false;
  }
  /** \brief Update equality constraint matrix incrementally.
      \param A new equality constraint matrix
      \return true if update successful
  */
  virtual bool updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A) { return false; }
  /** \brief Update equality constraint vector incrementally.
      \param b new equality constraint vector
      \return true if update successful
  */
  virtual bool updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b) { return false; }
  /** \brief Solve QP incrementally (after updates).
      \return solution vector
  */
  virtual Eigen::VectorXd solveIncremental()
  {
    // Fallback: if not implemented, return empty vector to indicate failure
    return Eigen::VectorXd::Zero(0);
  }
  /** \brief Get QP solver type. */
  inline QpSolverType type() const { return type_; }
  /** \brief Get whether it failed to solve the QP. */
  inline bool solveFailed() const { return solve_failed_; }
  /** \brief Get parameter manager interface.
      \return parameter manager shared pointer (returns nullptr for solvers that don't support
     parameters)
  */
  virtual std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const { return nullptr; }

protected:
  /** \brief QP solver type. */
  QpSolverType type_ = QpSolverType::Uninitialized;

  /** \brief Whether it failed to solve the QP. */
  bool solve_failed_ = false;
};
#if ENABLE_QLD
/** \brief QP solver QLD. */
class QpSolverQld : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverQld();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

protected:
  std::unique_ptr<Eigen::QLDDirect> qld_;
};
#endif

#if ENABLE_QUADPROG
/** \brief QP solver QuadProg. */
class QpSolverQuadprog : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverQuadprog();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

protected:
  std::unique_ptr<Eigen::QuadProgDense> quadprog_;
};
#endif

#if ENABLE_LSSOL
/** \brief QP solver LSSOL. */
class QpSolverLssol : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverLssol();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

protected:
  std::unique_ptr<Eigen::LSSOL_QP> lssol_;
};
#endif

#if ENABLE_JRLQP
/** \brief QP solver JRLQP. */
class QpSolverJrlqp : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverJrlqp();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

protected:
  std::unique_ptr<jrl::qp::GoldfarbIdnaniSolver> jrlqp_;
};
#endif

#if ENABLE_QPOASES
/** \brief QP solver qpOASES.
    \todo Support an efficient interface (QProblemB) dedicated to QP with only box constraints.
*/
class QpSolverQpoases : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverQpoases();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

  /** \brief Solve QP with both-sided inequality constraints. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper,
    const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();

public:
  int n_wsr_ = 10000;

  /** \brief Whether to initialize each time instead of doing a warm start.

      \note Warm start did not give good results.
  */
  bool force_initialize_ = true;

protected:
  std::unique_ptr<qpOASES::SQProblem> qpoases_;
  struct QpoasesParameters
  {
    int max_iter = 10;
    double termination_tolerance = 1e-2;
    double bound_tolerance = 1e-4;
    bool enable_cholesky_refactorisation = true;
    bool enable_regularisation = true;
    bool use_warm_start = true;
    int num_refinement_steps = 1;
    double eps_regularisation = 1e-6;  // Scaling factor for Hessian regularisation
    int num_regularisation_steps = 0;  // Maximum number of successive regularisation steps (0=automatic)
  };
  QpoasesParameters qpoases_params_;
  double solve_time_us_ = 0;  // [us] Pure solve time

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::shared_ptr<tam::tsl::ValueLogger> logger_ = std::make_shared<tam::tsl::ValueLogger>();
  std::size_t previous_param_state_hash_ = 0;
};
#endif

#if ENABLE_OSQP
/** \brief QP solver OSQP.
    \todo Set without going through a dense matrix.
 */
class QpSolverOsqp : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverOsqp();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

  /** \brief Solve QP with both-sided inequality constraints. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper,
    const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

  /** \brief Get solver status. */
  int getSolverStatus() const;

  /** \brief Get iteration count. */
  int getIterationCount() const;

  /** \brief Get primal residual. */
  double getPrimalResidual() const;

  /** \brief Get dual residual. */
  double getDualResidual() const;
  // ===== INCREMENTAL UPDATE IMPLEMENTATION =====
  bool supportsIncrementalUpdate() const override { return true; }
  bool updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q) override;
  bool updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c) override;
  bool updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C) override;
  bool updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d) override;
  bool updateInequalityVectorBothSide(
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper) override;
  bool updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A) override;
  bool updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b) override;
  Eigen::VectorXd solveIncremental() override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();

public:
  /** \brief Whether to initialize each time instead of doing a warm start.

      \note Warm start did not give good results.
  */
  bool force_initialize_ = true;

protected:
  std::unique_ptr<OsqpEigen::Solver> osqp_;

  Eigen::SparseMatrix<double> Q_sparse_;
  Eigen::VectorXd c_;
  Eigen::SparseMatrix<double> AC_with_bound_sparse_;
  Eigen::VectorXd bd_with_bound_min_;
  Eigen::VectorXd bd_with_bound_max_;

  // Dimension tracking for incremental updates
  int dim_var_ = 0;
  int dim_eq_ = 0;
  int dim_ineq_ = 0;

  double sparse_duration_ = 0;  // [ms]
  double solve_time_us_ = 0;    // [us] Pure solve time

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::shared_ptr<tam::tsl::ValueLogger> logger_ = std::make_shared<tam::tsl::ValueLogger>();
  std::size_t previous_param_state_hash_ = 0;
  // OSQP solver parameters
  struct OsqpParameters
  {
    int max_iter = 60;
    double abs_tolerance = 1e-3;
    double rel_tolerance = 1e-3;
    double alpha = 1.6;
    bool verbose = false;
    int scaling = 10;
    int adaptive_rho_interval = 0;
    bool polish = false;
    double time_limit = 0.0;
    int check_termination = 0;
  } osqp_params_;
  // Solver status tracking
  // int osqp_iterations_ = 0;
  // double osqp_primal_res_ = 0.0;
  // double osqp_dual_res_ = 0.0;
};
#endif

#if ENABLE_NASOQ
/** \brief QP solver NASOQ.
    \todo Set without going through a dense matrix.
*/
class QpSolverNasoq : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverNasoq();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();

protected:
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> Q_sparse_;
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_sparse_;
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> C_with_bound_sparse_;

  double sparse_duration_ = 0;  // [ms]
  struct NasoqParameters
  {
    int max_iter = 10;
    double eps_abs = 1e-5;
    double eps_rel = 1e-5;
    double regularization = 1e-9;
    std::string nasoq_variant = "auto";  //"tune", "fixed", "auto"
  };
  NasoqParameters nasoq_params_;
  double solve_time_us_ = 0;  // [us] Pure solve time

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::shared_ptr<tam::tsl::ValueLogger> logger_ = std::make_shared<tam::tsl::ValueLogger>();
  std::size_t previous_param_state_hash_ = 0;
};
#endif

#if ENABLE_HPIPM
/** \brief QP solver HPIPM. */
class QpSolverHpipm : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverHpipm();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

  /** \brief Solve QP with both-sided inequality constraints. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper,
    const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;
  // ===== INCREMENTAL UPDATE IMPLEMENTATION =====
  bool supportsIncrementalUpdate() const override { return true; }
  bool updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q) override;
  bool updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c) override;
  bool updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C) override;
  bool updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d) override;
  bool updateInequalityVectorBothSide(
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper) override;
  bool updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A) override;
  bool updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b) override;
  Eigen::VectorXd solveIncremental() override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();

public:
  /** \brief Maximum limits of inequality bounds.

      \note Setting very large values for inequality bounds will result in NaN or incorrect
     solutions. std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max(), and
     even 1e20 have this problem.
  */
  double bound_limit_ = 1e10;

protected:
  std::unique_ptr<struct d_dense_qp_dim> qp_dim_;
  std::unique_ptr<struct d_dense_qp> qp_;
  std::unique_ptr<struct d_dense_qp_sol> qp_sol_;
  std::unique_ptr<struct d_dense_qp_ipm_arg> ipm_arg_;
  std::unique_ptr<struct d_dense_qp_ipm_ws> ipm_ws_;

  std::unique_ptr<uint8_t[]> qp_dim_mem_ = nullptr;
  std::unique_ptr<uint8_t[]> qp_mem_ = nullptr;
  std::unique_ptr<uint8_t[]> qp_sol_mem_ = nullptr;
  std::unique_ptr<uint8_t[]> ipm_arg_mem_ = nullptr;
  std::unique_ptr<uint8_t[]> ipm_ws_mem_ = nullptr;

  std::unique_ptr<double[]> opt_x_mem_ = nullptr;

  // Dimension tracking for incremental updates and initialization check
  bool initialized_ = false;
  int dim_var_ = 0;
  int dim_eq_ = 0;
  int dim_ineq_ = 0;
  struct HpipmParameters
  {
    int max_iter = 10;  // Maximum number of iterations
    double tol_stat = 1e-3;
    double tol_eq = 1e-3;
    double tol_ineq = 1e-3;
    double tol_comp = 1e-3;
    int warm_start = 1;
    int pred_corr = 1;
    int cond_pred_corr = 1;
    int split_step = 0;
  };
  HpipmParameters hpipm_params_;
  double solve_time_us_ = 0;  // [us] Pure solve time

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::shared_ptr<tam::tsl::ValueLogger> logger_ = std::make_shared<tam::tsl::ValueLogger>();
  std::size_t previous_param_state_hash_ = 0;
};
#endif

#if ENABLE_PROXQP
/** \brief QP solver PROXQP. */
class QpSolverProxqp : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverProxqp();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;

  /** \brief Solve QP with both-sided inequality constraints. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper,
    const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();
  // ===== INCREMENTAL UPDATE IMPLEMENTATION =====
  bool supportsIncrementalUpdate() const override { return true; }
  bool updateObjectiveMatrix(Eigen::Ref<Eigen::MatrixXd> Q) override;
  bool updateObjectiveVector(const Eigen::Ref<const Eigen::VectorXd> & c) override;
  bool updateInequalityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & C) override;
  bool updateInequalityVector(const Eigen::Ref<const Eigen::VectorXd> & d) override;
  bool updateInequalityVectorBothSide(
    const Eigen::Ref<const Eigen::VectorXd> & d_lower,
    const Eigen::Ref<const Eigen::VectorXd> & d_upper) override;
  bool updateEqualityMatrix(const Eigen::Ref<const Eigen::MatrixXd> & A) override;
  bool updateEqualityVector(const Eigen::Ref<const Eigen::VectorXd> & b) override;
  Eigen::VectorXd solveIncremental() override;

protected:
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> proxqp_;
  struct ProxqpParameters
  {
    double eps_abs = 1e-6;  // Absolute tolerance for stopping criterion (||residual|| <= eps_abs)
    double eps_rel =
      1e-6;  // Relative tolerance for stopping criterion (||residual|| / ||rhs|| <= eps_rel)
    int max_iter = 50;
    bool verbose = false;  // Whether to print solver progress during iterations
    bool warm_start = true;
    bool compute_timings =
      false;  // Whether to compute timing information (useful for benchmarking)
    bool check_duality_gap = false;  // Whether to check duality gap at the end of the solve
  };
  ProxqpParameters proxqp_params_;
  double solve_time_us_ = 0;  // [us] Pure solve time

  // Dimension tracking for incremental updates
  int dim_var_ = 0;
  int dim_eq_ = 0;
  int dim_ineq_ = 0;

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::shared_ptr<tam::tsl::ValueLogger> logger_ = std::make_shared<tam::tsl::ValueLogger>();
  std::size_t previous_param_state_hash_ = 0;
};
#endif

#if ENABLE_QPMAD
/** \brief QP solver QPMAD. */
class QpSolverQpmad : public QpSolver
{
public:
  /** \brief Constructor. */
  QpSolverQpmad();

  /** \brief Solve QP. */
  virtual Eigen::VectorXd solve(
    int dim_var, int dim_eq, int dim_ineq, Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::VectorXd> & c, const Eigen::Ref<const Eigen::MatrixXd> & A,
    const Eigen::Ref<const Eigen::VectorXd> & b, const Eigen::Ref<const Eigen::MatrixXd> & C,
    const Eigen::Ref<const Eigen::VectorXd> & d, const Eigen::Ref<const Eigen::VectorXd> & x_min,
    const Eigen::Ref<const Eigen::VectorXd> & x_max) override;
  /** \brief Get parameter manager interface. */
  std::shared_ptr<tam::pmg::ParamValueManager> getParamHandler() const override
  {
    return param_manager_;
  }
  /** \brief Declare and update parameters from parameter manager. */
  void declare_and_update_parameters();

protected:
  std::unique_ptr<qpmad::Solver> qpmad_;

  std::shared_ptr<tam::pmg::ParamValueManager> param_manager_ =
    std::make_shared<tam::pmg::ParamValueManager>();
  std::size_t previous_param_state_hash_ = 0;
};
#endif

/** \brief Get one QP solver type that is enabled.

    Checks whether each QP solver is enabled in the order of definition in QpSolverType and returns
   the first one that is enabled.
 */
QpSolverType getAnyQpSolverType();

/** \brief Check whether QP solver is enabled.
    \param qp_solver_type QP solver type
 */
bool isQpSolverEnabled(const QpSolverType & qp_solver_type);

/** \brief Allocate the specified QP solver.
    \param qp_solver_type QP solver type
 */
std::shared_ptr<QpSolver> allocateQpSolver(const QpSolverType & qp_solver_type);
}  // namespace QpSolverCollection
