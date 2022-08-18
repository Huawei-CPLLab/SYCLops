configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SYCLOPS_TEST_DEPS
  clang
  FileCheck
  sycl
  syclops
)
add_lit_testsuite(check-syclops "Running the SYCLops tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SYCLOPS_TEST_DEPS}
    ARGS -v
)

set_target_properties(check-syclops PROPERTIES FOLDER "Tests")