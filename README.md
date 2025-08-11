This is the branch for ROS2; use the [ros1](https://github.com/isri-aist/QpSolverCollection/tree/ros1) branch for ROS1.

# [QpSolverCollection](https://github.com/isri-aist/QpSolverCollection)
Unified C++ interface for quadratic programming solvers

[![CI](https://github.com/isri-aist/QpSolverCollection/actions/workflows/ci.yaml/badge.svg)](https://github.com/isri-aist/QpSolverCollection/actions/workflows/ci.yaml)
[![Documentation](https://img.shields.io/badge/doxygen-online-brightgreen?logo=read-the-docs&style=flat)](https://isri-aist.github.io/QpSolverCollection/)

## Features
- Unified C++ interface to many QP solvers
- Can be built as a standalone package or ROS package
- High portability decoupled from each QP solver by [Pimpl technique](https://en.cppreference.com/w/cpp/language/pimpl)

## Supported QP solvers
- [QLD](https://github.com/jrl-umi3218/eigen-qld)
- [QuadProg](https://github.com/jrl-umi3218/eigen-quadprog)
- [JRLQP](https://github.com/jrl-umi3218/jrl-qp)
- [qpOASES](https://github.com/coin-or/qpOASES)
- [OSQP](https://osqp.org/)
- [NASOQ](https://nasoq.github.io/)
- [HPIPM](https://github.com/giaf/hpipm)
- [ProxQP](https://github.com/Simple-Robotics/proxsuite)
- [qpmad](https://github.com/asherikov/qpmad)
- [LSSOL](https://gite.lirmm.fr/multi-contact/eigen-lssol) (private)

## Installation

### Installation procedure
It is assumed that ROS is installed.

1. Install the QP solver you wish to use according to [this section](https://github.com/isri-aist/QpSolverCollection#qp-solver-installation). You can skip installing QP solvers that you do not use.

2. Setup colcon workspace.
```bash
$ mkdir -p ~/ros/ws_qp_solver_collection/src
$ cd ~/ros/ws_qp_solver_collection
$ wstool init src
$ wstool set -t src isri-aist/QpSolverCollection git@github.com:isri-aist/QpSolverCollection.git --git -y
$ wstool update -t src
```

> `wstool` can be installed with `apt install python3-wstool` or `pip install wstool`.

3. Install dependent packages.
```bash
$ source /opt/ros/${ROS_DISTRO}/setup.bash
$ rosdep install -y -r --from-paths src --ignore-src
```

4. Build a package.
```bash
$ cd ~/ros/ws_qp_solver_collection
$ colcon build --packages-select qp_solver_collection --merge-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo <qp-solver-flags>
$ colcon test --merge-install --packages-select qp_solver_collection # [optional] to compile and run tests 
```
See [this section](https://github.com/isri-aist/QpSolverCollection#qp-solver-installation) for `<qp-solver-flags>`.

### QP solver installation
As all supported QP solvers are installed in [CI](https://github.com/isri-aist/QpSolverCollection/blob/master/.github/workflows/ci.yaml), please refer to the installation procedure.  
Please refer to the license specified in each QP solver when using it.

#### QLD
Install [eigen-qld](https://github.com/jrl-umi3218/eigen-qld).  
Add `-DENABLE_QLD=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### QuadProg
Install [eigen-quadprog](https://github.com/jrl-umi3218/eigen-quadprog).  
Add `-DENABLE_QUADPROG=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### JRLQP
Install master branch of [jrl-qp](https://github.com/jrl-umi3218/jrl-qp).
Add `-DENABLE_JRLQP=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### qpOASES
Install master branch of [qpOASES](https://github.com/coin-or/qpOASES) with `-DBUILD_SHARED_LIBS=ON` cmake option.  
Add `-DENABLE_QPOASES=ON` to the catkin build command (i.e., `<qp-solver-flags>`).
Also, add `-DQPOASES_INCLUDE_DIR=<path to qpOASES.hpp>` and `-DQPOASES_LIBRARY_DIR=<path to libqpOASES.so>` to the catkin build command.

#### OSQP
Install master branch of [osqp](https://github.com/osqp/osqp) and [osqp-eigen](https://github.com/robotology/osqp-eigen).  
Add `-DENABLE_OSQP=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### NASOQ
Install [cmake-install branch](https://github.com/mmurooka/nasoq/tree/cmake-install) of nasoq.  
Add `-DENABLE_NASOQ=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### HPIPM
Install master branch of [blasfeo](https://github.com/giaf/blasfeo) and [hpipm](https://github.com/giaf/hpipm).  
For hpipm installation, it's recommended to checkout to commit [00c2a084e059e2e1b79877f668e282d0c4282110](https://github.com/giaf/hpipm/commit/00c2a084e059e2e1b79877f668e282d0c4282110) as documented.  
Add `/opt/blasfeo/lib` and `/opt/hpipm/lib` to the environment variable `LD_LIBRARY_PATH`.  
Add `-DENABLE_HPIPM=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### ProxQP
Install main branch of [proxsuite](https://github.com/Simple-Robotics/proxsuite).  
Add `-DENABLE_PROXQP=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

**Note**: ProxQP requires the Simde library. If you encounter a CMake error about missing `Simde_INCLUDE_DIR`, clone [simde](https://github.com/simd-everywhere/simde) and add `-DSimde_INCLUDE_DIR=<path-to-simde>` to your build command.

#### qpmad
Install master branch of [qpmad](https://github.com/asherikov/qpmad).  
Add `-DENABLE_QPMAD=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

#### LSSOL (private)
Install [eigen-lssol](https://gite.lirmm.fr/multi-contact/eigen-lssol).  
Add `-DENABLE_LSSOL=ON` to the catkin build command (i.e., `<qp-solver-flags>`).

## How to use
See [documentation](https://isri-aist.github.io/QpSolverCollection/doxygen/classQpSolverCollection_1_1QpSolver.html) and [test](https://github.com/isri-aist/QpSolverCollection/blob/master/tests/TestSampleQP.cpp) for examples of solving QP problems.

The following is a simple sample.
```cpp
// sample.cpp

#include <qp_solver_collection/QpSolverCollection.h>

int main()
{
  int dim_var = 2;
  int dim_eq = 1;
  int dim_ineq = 0;
  QpSolverCollection::QpCoeff qp_coeff;
  qp_coeff.setup(dim_var, dim_eq, dim_ineq);
  qp_coeff.obj_mat_ << 2.0, 0.5, 0.5, 1.0;
  qp_coeff.obj_vec_ << 1.0, 1.0;
  qp_coeff.eq_mat_ << 1.0, 1.0;
  qp_coeff.eq_vec_ << 1.0;
  qp_coeff.x_min_.setZero();
  qp_coeff.x_max_.setConstant(1000.0);

  auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::Any);
  Eigen::VectorXd solution = qp_solver->solve(qp_coeff);
  std::cout << "solution: " << solution.transpose() << std::endl;

  return 0;
}
```

In addition to building a sample in a catkin package, you can also build it standalone as follows.
```bash
$ g++ sample.cpp `pkg-config --cflags qp_solver_collection` `pkg-config --libs qp_solver_collection`
```

## Known Issues and Troubleshooting

#### Simde dependency for ProxQP
If building with ProxQP enabled fails with `Could NOT find Simde (missing: Simde_INCLUDE_DIR)`, you need to:
1. Clone the simde library: `git clone https://github.com/simd-everywhere/simde.git`
2. Add the include path to your build command: `-DSimde_INCLUDE_DIR=/path/to/simde`

#### NASOQ CMake configuration missing
If you encounter `Could NOT find nasoq (missing: nasoq_LIBRARY nasoq_EIGEN_INCLUDE_DIR nasoq_MAIN_INCLUDE_DIR)`:
1. The official NASOQ repository doesn't provide `make install` by default
2. You need to modify the paths in `cmake/Findnasoq.cmake`:
   - Update `nasoq_EIGEN_INCLUDE_DIR` paths to point to your `nasoq/eigen_interface/include` directory
   - Update `nasoq_MAIN_INCLUDE_DIR` paths to point to your `nasoq/include` directory  
   - Update `nasoq_LIBRARY` paths to point to your `nasoq/lib` directory
3. Alternatively, disable NASOQ if you don't need it: `-DENABLE_NASOQ=OFF`

#### HPIPM CMake configuration missing
If you encounter `Could NOT find hpipm (missing: hpipm_LIBRARY hpipm_INCLUDE_DIR)`:
1. HPIPM does provide `make install`, but installs to local directories (not `/usr/local` by default)
2. You need to modify the paths in `cmake/Findhpipm.cmake`:
   - Update `hpipm_INCLUDE_DIR` paths to point to your `hpipm/include` directory
   - Update `hpipm_LIBRARY` paths to point to your `hpipm/lib` directory
3. Alternatively, disable HPIPM if you don't need it: `-DENABLE_HPIPM=OFF`
4. Recommended: build HPIPM statically to avoid runtime loader issues: `-make static_library`

#### BLASFEO CMake configuration missing（HPIPM Dependency）  
If you encounter `Could NOT find blasfeo (missing: blasfeo_LIBRARY blasfeo_INCLUDE_DIR)`:
1. BLASFEO does provide `make install`, but installs to local directories (not `/usr/local` by default)
2. You need to modify the paths in `cmake/Findblasfeo.cmake`:
   - Update `blasfeo_INCLUDE_DIR` paths to point to your `blasfeo/include` directory
   - Update `blasfeo_LIBRARY` paths to point to your `blasfeo/lib` directory
3. Alternatively, disable BLASFEO-dependent solvers if you don't need them: `-DENABLE_HPIPM=OFF`
4. Recommended: build HPIPM statically to avoid runtime loader issues: `make static_library`

Example build command with all fixes:
```bash
colcon build --packages-select qp_solver_collection --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_OSQP=ON -DENABLE_QPOASES=ON -DENABLE_HPIPM=ON -DENABLE_NASOQ=ON -DENABLE_PROXQP=ON -DSimde_INCLUDE_DIR=/home/user/ocp_solver/simde qp_solver_collection
```
