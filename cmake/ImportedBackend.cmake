# Use only the matching bundled headers and a recorded backend DLL family.
foreach(component ggml-base ggml-cpu)
    add_library(${component} SHARED IMPORTED GLOBAL)
    set_target_properties(${component} PROPERTIES
        IMPORTED_IMPLIB "${QW_BACKEND_BUILD}/third_party/ggml/src/${component}.lib"
        IMPORTED_LOCATION "${QW_BACKEND_BUILD}/bin/${component}.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include")
endforeach()
set_property(TARGET ggml-cpu PROPERTY INTERFACE_LINK_LIBRARIES ggml-base)
set_property(TARGET ggml-base PROPERTY INTERFACE_COMPILE_DEFINITIONS GGML_SHARED)
set_property(TARGET ggml-cpu PROPERTY INTERFACE_COMPILE_DEFINITIONS GGML_BACKEND_SHARED)

# Reuse upstream registration/loader code, not the old dispatcher DLL: that
# dispatcher has an import-time RPC dependency even when no RPC worker is used.
add_library(ggml SHARED
    third_party/ggml/src/ggml-backend-dl.cpp
    third_party/ggml/src/ggml-backend-reg.cpp)
target_include_directories(ggml PUBLIC third_party/ggml/include PRIVATE third_party/ggml/src)
target_compile_definitions(ggml PRIVATE GGML_BUILD GGML_USE_CPU PUBLIC GGML_SHARED)
target_link_libraries(ggml PUBLIC ggml-base ggml-cpu)
if(QW_CUDA)
    add_library(ggml-cuda SHARED IMPORTED GLOBAL)
    set_target_properties(ggml-cuda PROPERTIES
        IMPORTED_IMPLIB "${QW_BACKEND_BUILD}/third_party/ggml/src/ggml-cuda/ggml-cuda.lib"
        IMPORTED_LOCATION "${QW_BACKEND_BUILD}/bin/ggml-cuda.dll"
        INTERFACE_COMPILE_DEFINITIONS GGML_BACKEND_SHARED)
    target_compile_definitions(ggml PRIVATE GGML_USE_CUDA)
    target_link_libraries(ggml PUBLIC ggml-cuda)
endif()
