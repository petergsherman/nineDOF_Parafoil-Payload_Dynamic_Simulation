// =============================================================================
// parafoil_bindings.cpp
// pybind11 bindings: exposes ParafoilEnv to Python.
//
// After building, import in Python as:
//   import parafoil_cpp
//   env = parafoil_cpp.ParafoilEnv(params, target_x, target_y)
//   obs = env.reset(seed=42)
//   obs, reward, done, info = env.step([0.0, 0.0])
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ParafoilEnv.h"
#include "nineDOF_Parameters.h"

namespace py = pybind11;

PYBIND11_MODULE(parafoil_cpp, m) {
    m.doc() = "9DOF Parafoil RL environment - C++ backend";

    // -------------------------------------------------------------------------
    // Expose SystemParameters so Python can configure them
    // -------------------------------------------------------------------------
    py::class_<SystemParameters>(m, "SystemParameters")
        .def(py::init<>())
        // Masses and geometry
        .def_readwrite("m_parafoil",      &SystemParameters::m_parafoil)
        .def_readwrite("m_cradle",        &SystemParameters::m_cradle)
        .def_readwrite("A_parafoil",      &SystemParameters::A_parafoil)
        .def_readwrite("A_cradle",        &SystemParameters::A_cradle)
        .def_readwrite("cbar_parafoil",   &SystemParameters::cbar_parafoil)
        .def_readwrite("bbar_parafoil",   &SystemParameters::bbar_parafoil)
        .def_readwrite("dbar",            &SystemParameters::dbar)
        .def_readwrite("deadband",        &SystemParameters::deadband)
        .def_readwrite("CD_cradle",       &SystemParameters::CD_cradle)
        // Parafoil inertia
        .def_readwrite("PIXX", &SystemParameters::PIXX)
        .def_readwrite("PIYY", &SystemParameters::PIYY)
        .def_readwrite("PIZZ", &SystemParameters::PIZZ)
        // Control parameters
        .def_readwrite("NOM_INCIDENCE",   &SystemParameters::NOM_INCIDENCE)
        .def_readwrite("NOM_BRAKE",       &SystemParameters::NOM_BRAKE)
        .def_readwrite("TAU",             &SystemParameters::TAU)
        .def_readwrite("KGIMBAL",         &SystemParameters::KGIMBAL)
        .def_readwrite("CGIMBAL",         &SystemParameters::CGIMBAL)
        // Physical constants
        .def_readwrite("GRAVITY",         &SystemParameters::GRAVITY);

    // -------------------------------------------------------------------------
    // Expose DomainRandomConfig
    // -------------------------------------------------------------------------
    py::class_<DomainRandomConfig>(m, "DomainRandomConfig")
        .def(py::init<>())
        .def_readwrite("pos_x_range",   &DomainRandomConfig::pos_x_range)
        .def_readwrite("pos_y_range",   &DomainRandomConfig::pos_y_range)
        .def_readwrite("alt_min",       &DomainRandomConfig::alt_min)
        .def_readwrite("alt_max",       &DomainRandomConfig::alt_max)
        .def_readwrite("heading_range", &DomainRandomConfig::heading_range)
        .def_readwrite("u_nominal",     &DomainRandomConfig::u_nominal)
        .def_readwrite("u_noise",       &DomainRandomConfig::u_noise)
        .def_readwrite("w_nominal",     &DomainRandomConfig::w_nominal)
        .def_readwrite("w_noise",       &DomainRandomConfig::w_noise)
        .def_readwrite("wind_x_range",  &DomainRandomConfig::wind_x_range)
        .def_readwrite("wind_y_range",  &DomainRandomConfig::wind_y_range)
        .def_readwrite("theta_noise",   &DomainRandomConfig::theta_noise)
        .def_readwrite("phi_noise",     &DomainRandomConfig::phi_noise);

    // -------------------------------------------------------------------------
    // Expose StepInfo
    // -------------------------------------------------------------------------
    py::class_<StepInfo>(m, "StepInfo")
        .def(py::init<>())
        .def_readwrite("distance_to_target", &StepInfo::distance_to_target)
        .def_readwrite("altitude",           &StepInfo::altitude)
        .def_readwrite("heading",            &StepInfo::heading)
        .def_readwrite("airspeed",           &StepInfo::airspeed)
        .def_readwrite("hit_ground",         &StepInfo::hit_ground)
        .def_readwrite("diverged",           &StepInfo::diverged)
        .def_readwrite("step_count",         &StepInfo::step_count)
        .def_readwrite("landing_error",      &StepInfo::landing_error);

    // -------------------------------------------------------------------------
    // Expose ParafoilEnv - the main RL environment backend
    // -------------------------------------------------------------------------
    py::class_<ParafoilEnv>(m, "ParafoilEnv")
        .def(py::init<const SystemParameters&, double, double, double, double>(),
             py::arg("params"),
             py::arg("target_x")    = 0.0,
             py::arg("target_y")    = 0.0,
             py::arg("dt_physics")  = 0.01,
             py::arg("dt_action")   = 0.1,
             R"doc(
             Create a parafoil RL environment.

             Args:
                 params:     SystemParameters struct with vehicle configuration.
                 target_x:   Landing target X coordinate (m, inertial frame).
                 target_y:   Landing target Y coordinate (m, inertial frame).
                 dt_physics: Physics integration timestep (s). Default 0.01.
                 dt_action:  Time between RL actions (s). Default 0.1.
                             (steps_per_action = dt_action / dt_physics)
             )doc")

        // reset() - returns numpy array for observation
        .def("reset",
             [](ParafoilEnv& self, int seed) -> py::array_t<double> {
                 auto obs = self.reset(seed);
                 return py::array_t<double>(obs.size(), obs.data());
             },
             py::arg("seed") = -1,
             "Reset the environment and return initial observation.")

        // reset() with fixed state (for evaluation)
        .def("reset_fixed",
             [](ParafoilEnv& self, py::array_t<double> state_arr) -> py::array_t<double> {
                 auto buf = state_arr.request();
                 if (buf.size != 18) throw std::runtime_error("State must have 18 elements");
                 State state;
                 double* ptr = static_cast<double*>(buf.ptr);
                 for (int i = 0; i < 18; i++) state[i] = ptr[i];
                 auto obs = self.reset(-1, &state);
                 return py::array_t<double>(obs.size(), obs.data());
             },
             "Reset the environment with a fixed initial state (18-element array).")

        // step() - returns (obs, reward, done, info_dict)
        .def("step",
             [](ParafoilEnv& self, py::array_t<double> action_arr)
                 -> std::tuple<py::array_t<double>, double, bool, py::dict>
             {
                 auto buf = action_arr.request();
                 if (buf.size != ACT_SIZE) {
                     throw std::runtime_error("Action must have 2 elements: [deltaL, deltaR]");
                 }
                 double* ptr = static_cast<double*>(buf.ptr);
                 std::array<double, ACT_SIZE> action = {ptr[0], ptr[1]};

                 auto [obs, reward, done, info] = self.step(action);
                 py::array_t<double> obs_arr(obs.size(), obs.data());

                 // Build info dict from the C++ StepInfo struct
                 py::dict info_dict;
                 info_dict["distance_to_target"] = info.distance_to_target;
                 info_dict["altitude"]           = info.altitude;
                 info_dict["heading"]            = info.heading;
                 info_dict["airspeed"]           = info.airspeed;
                 info_dict["hit_ground"]         = info.hit_ground;
                 info_dict["diverged"]           = info.diverged;
                 info_dict["step_count"]         = info.step_count;
                 info_dict["landing_error"]      = info.landing_error;

                 return {obs_arr, reward, done, info_dict};
             },
             "Step the environment. Action: [deltaL, deltaR] in [0, 0.94].")

        // Observation / state accessors
        .def("get_observation",
             [](ParafoilEnv& self) -> py::array_t<double> {
                 auto obs = self.getObservation();
                 return py::array_t<double>(obs.size(), obs.data());
             })
        .def("get_state",
             [](ParafoilEnv& self) -> py::array_t<double> {
                 auto state = self.getState();
                 return py::array_t<double>(state.size(), state.data());
             })
        .def("is_done",     &ParafoilEnv::isDone)
        .def("get_time",    &ParafoilEnv::getTime)
        .def("get_info",
             [](ParafoilEnv& self) -> py::dict {
                 StepInfo info = self.getInfo();
                 py::dict d;
                 d["distance_to_target"] = info.distance_to_target;
                 d["altitude"]           = info.altitude;
                 d["heading"]            = info.heading;
                 d["airspeed"]           = info.airspeed;
                 d["hit_ground"]         = info.hit_ground;
                 d["diverged"]           = info.diverged;
                 d["step_count"]         = info.step_count;
                 d["landing_error"]      = info.landing_error;
                 return d;
             })

        // Configuration
        .def("set_target",              &ParafoilEnv::setTarget)
        .def("set_domain_random_config",&ParafoilEnv::setDomainRandomConfig)
        .def("set_max_episode_time",    &ParafoilEnv::setMaxEpisodeTime)
        .def("set_wind",                &ParafoilEnv::setWind,
             py::arg("vx"), py::arg("vy"), py::arg("vz") = 0.0)

        // Expose observation size as class attribute
        .def_property_readonly_static("OBS_SIZE",
             [](py::object) { return OBS_SIZE; })
        .def_property_readonly_static("ACT_SIZE",
             [](py::object) { return ACT_SIZE; })

        // Expose normalization bounds for gym space construction
        .def_property_readonly_static("OBS_POS_NORM",
             [](py::object) { return ParafoilEnv::OBS_POS_NORM; })
        .def_property_readonly_static("OBS_ALT_NORM",
             [](py::object) { return ParafoilEnv::OBS_ALT_NORM; })
        .def_property_readonly_static("OBS_VEL_NORM",
             [](py::object) { return ParafoilEnv::OBS_VEL_NORM; });

    // -------------------------------------------------------------------------
    // Module-level constants
    // -------------------------------------------------------------------------
    m.attr("OBS_SIZE") = OBS_SIZE;
    m.attr("ACT_SIZE") = ACT_SIZE;
    m.attr("MAX_BRAKE") = 0.94;
}