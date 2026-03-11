# Install script for directory: C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/parafoil_rl")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Strawberry/c/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env/../python_env/parafoil_cpp.cp313-win_amd64.pyd")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env/../python_env" TYPE MODULE FILES "C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/build/parafoil_cpp.cp313-win_amd64.pyd")
  if(EXISTS "$ENV{DESTDIR}/C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env/../python_env/parafoil_cpp.cp313-win_amd64.pyd" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env/../python_env/parafoil_cpp.cp313-win_amd64.pyd")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "C:/Strawberry/c/bin/strip.exe" "$ENV{DESTDIR}/C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/cpp_env/../python_env/parafoil_cpp.cp313-win_amd64.pyd")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/build/CMakeFiles/parafoil_cpp.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/Pete/Desktop/nineDOF_Parafoil-Payload_Dynamic_Simulation/parafoil_rl/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
