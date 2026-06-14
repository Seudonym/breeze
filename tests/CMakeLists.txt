add_executable(unit_tests
  test_tensor.cu
)

target_link_libraries(unit_tests
  PRIVATE
    breeze
    GTest::gtest_main
)

add_test(NAME unit_tests COMMAND unit_tests)
