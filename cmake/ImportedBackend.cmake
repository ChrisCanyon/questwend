# Use only the matching bundled headers and a recorded backend DLL family.
foreach(component ggml-base ggml-cpu ggml)
    add_library(${component} SHARED IMPORTED GLOBAL)
    set_target_properties(${component} PROPERTIES
        IMPORTED_IMPLIB "${QW_BACKEND_BUILD}/third_party/ggml/src/${component}.lib"
        IMPORTED_LOCATION "${QW_BACKEND_BUILD}/bin/${component}.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include")
endforeach()
set_property(TARGET ggml-cpu PROPERTY INTERFACE_LINK_LIBRARIES ggml-base)
set_property(TARGET ggml PROPERTY INTERFACE_LINK_LIBRARIES "ggml-base;ggml-cpu")
