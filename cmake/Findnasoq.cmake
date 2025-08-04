# Findnasoq.cmake - Find nasoq library
find_path(nasoq_EIGEN_INCLUDE_DIR
    NAMES nasoq/nasoq_eigen.h
    PATHS 
        ~/ocp_solver/nasoq/eigen_interface/include
        /usr/local/include
    DOC "nasoq eigen interface include directory"
)

find_path(nasoq_MAIN_INCLUDE_DIR
    NAMES nasoq/nasoq.h
    PATHS 
        ~/ocp_solver/nasoq/include
        /usr/local/include
    DOC "nasoq main include directory"
)

find_library(nasoq_LIBRARY
    NAMES nasoq
    PATHS 
        ~/ocp_solver/nasoq/lib
        /usr/local/lib
    DOC "nasoq library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nasoq DEFAULT_MSG
    nasoq_LIBRARY nasoq_EIGEN_INCLUDE_DIR nasoq_MAIN_INCLUDE_DIR
)

if(nasoq_FOUND)
    # Find required dependencies
    find_package(OpenMP REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    
    # Find LAPACKE specifically
    find_library(LAPACKE_LIBRARY NAMES lapacke)
    if(NOT LAPACKE_LIBRARY)
        message(WARNING "LAPACKE library not found, trying to use system LAPACK")
        set(LAPACKE_LIBRARY "")
    endif()
    
    # Find METIS
    find_library(METIS_LIBRARY NAMES metis)
    if(NOT METIS_LIBRARY)
        message(FATAL_ERROR "METIS library not found")
    endif()
    
    set(nasoq_LIBRARIES ${nasoq_LIBRARY})
    set(nasoq_INCLUDE_DIRS ${nasoq_EIGEN_INCLUDE_DIR} ${nasoq_MAIN_INCLUDE_DIR})
    
    if(NOT TARGET nasoq::nasoq)
        add_library(nasoq::nasoq STATIC IMPORTED)
        set_target_properties(nasoq::nasoq PROPERTIES
            IMPORTED_LOCATION "${nasoq_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${nasoq_EIGEN_INCLUDE_DIR};${nasoq_MAIN_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "OpenMP::OpenMP_CXX;${BLAS_LIBRARIES};${LAPACK_LIBRARIES};${LAPACKE_LIBRARY};${METIS_LIBRARY}"
        )
    endif()
    
    mark_as_advanced(nasoq_EIGEN_INCLUDE_DIR nasoq_MAIN_INCLUDE_DIR nasoq_LIBRARY)
endif()