# Findhpipm.cmake - Find hpipm library
find_path(hpipm_INCLUDE_DIR
    NAMES hpipm_d_ocp_qp.h
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../ocp_solver/hpipm/include
        ~/ocp_solver/hpipm/include
        /usr/local/include
    DOC "hpipm include directory"
)

find_library(hpipm_LIBRARY
    NAMES hpipm
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../ocp_solver/hpipm/lib
        ~/ocp_solver/hpipm/lib
        /usr/local/lib
    DOC "hpipm library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hpipm DEFAULT_MSG
    hpipm_LIBRARY hpipm_INCLUDE_DIR
)

if(hpipm_FOUND)
    set(hpipm_LIBRARIES ${hpipm_LIBRARY})
    set(hpipm_INCLUDE_DIRS ${hpipm_INCLUDE_DIR})
    
    if(NOT TARGET hpipm::hpipm)
        add_library(hpipm::hpipm STATIC IMPORTED)
        set_target_properties(hpipm::hpipm PROPERTIES
            IMPORTED_LOCATION "${hpipm_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${hpipm_INCLUDE_DIR}"
        )
    endif()
    
    mark_as_advanced(hpipm_INCLUDE_DIR hpipm_LIBRARY)
endif()