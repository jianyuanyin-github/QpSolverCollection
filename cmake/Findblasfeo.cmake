# Findblasfeo.cmake - Find blasfeo library
find_path(blasfeo_INCLUDE_DIR
    NAMES blasfeo_common.h
    PATHS 
        ~/ocp_solver/blasfeo/include
        /opt/blasfeo/include
    DOC "blasfeo include directory"
)

find_library(blasfeo_LIBRARY
    NAMES blasfeo
    PATHS 
        ~/ocp_solver/blasfeo/lib
        /opt/blasfeo/lib
    DOC "blasfeo library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blasfeo DEFAULT_MSG
    blasfeo_LIBRARY blasfeo_INCLUDE_DIR
)

if(blasfeo_FOUND)
    set(blasfeo_LIBRARIES ${blasfeo_LIBRARY})
    set(blasfeo_INCLUDE_DIRS ${blasfeo_INCLUDE_DIR})
    
    if(NOT TARGET blasfeo::blasfeo)
        add_library(blasfeo::blasfeo STATIC IMPORTED)
        set_target_properties(blasfeo::blasfeo PROPERTIES
            IMPORTED_LOCATION "${blasfeo_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${blasfeo_INCLUDE_DIR}"
        )
    endif()
    
    mark_as_advanced(blasfeo_INCLUDE_DIR blasfeo_LIBRARY)
endif()