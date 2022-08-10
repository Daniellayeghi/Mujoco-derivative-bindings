#include "functions.h"
#include "raw.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace mujoco::python {
namespace {

    using EigenMatrixXMap = Eigen::Map<Eigen::Matrix<double, -1, 1>>;

    void copy_data(const raw::MjModel *model, const raw::MjData *data_src, raw::MjData *data_cp) {
        data_cp->time = data_src->time;
        mju_copy(data_cp->qpos, data_src->qpos, model->nq);
        mju_copy(data_cp->qvel, data_src->qvel, model->nv);
        mju_copy(data_cp->qacc, data_src->qacc, model->nv);
        mju_copy(data_cp->qfrc_applied, data_src->qfrc_applied, model->nv);
        mju_copy(data_cp->xfrc_applied, data_src->xfrc_applied, 6 * model->nbody);
        mju_copy(data_cp->ctrl, data_src->ctrl, model->nu);
        mj_forward(model, data_cp);
    }


    enum class Wrt : int {
        Pos = 0, Vel = 1, Acc = 2, State = 3, Ctrl = 4
    };

    enum class Mode : int {
        Fwd = 0, Inv = 1
    };


    class MjDerivativeParams{
    public:
        MjDerivativeParams(double eps, const Wrt wrt, const Mode mode): m_eps(eps), m_wrt_id(wrt), m_mode_id(mode){};
        double m_eps = 1e-6;
        const Wrt m_wrt_id = Wrt::Ctrl;
        const Mode m_mode_id = Mode::Fwd;
    };


    // TODO add function ptr to finite difference against e.g. step forward inverse ...
    //  The argument type should match binding
    struct MjDataVecView {
        MjDataVecView(const MjModelWrapper &m, MjDataWrapper &d) :
                m_m(m.get()),
                m_d(d.get()),
                m_ctrl(m_d->ctrl, m_m->nu),
                m_qfrc_inverse(m_d->qfrc_inverse, m_m->nu),
                m_pos(m_d->qpos, m_m->nq),
                m_vel(m_d->qvel, m_m->nv),
                m_acc(m_d->qacc, m_m->nv),
                m_sens(m_d->sensordata, m_m->nsensordata) {}

        MjDataVecView(const raw::MjModel *m, raw::MjData *d) :
                m_m(m),
                m_d(d),
                m_ctrl(m_d->ctrl, m_m->nu),
                m_qfrc_inverse(m_d->qfrc_inverse, m_m->nu),
                m_pos(m_d->qpos, m_m->nq),
                m_vel(m_d->qvel, m_m->nv),
                m_acc(m_d->qacc, m_m->nv),
                m_sens(m_d->sensordata, m_m->nsensordata) {}


        const raw::MjModel *m_m;
        raw::MjData *m_d;
        Eigen::Map <Eigen::VectorXd> m_ctrl;
        Eigen::Map <Eigen::VectorXd> m_qfrc_inverse;
        Eigen::Map <Eigen::VectorXd> m_pos;
        Eigen::Map <Eigen::VectorXd> m_vel;
        Eigen::Map <Eigen::VectorXd> m_acc;
        Eigen::Map <Eigen::VectorXd> m_sens;
    };


    class MjDerivative {
    public:
        explicit MjDerivative(const MjModelWrapper &m, const MjDerivativeParams& params) :
        m_d(mj_makeData(m.get())), m_ed(m.get(), m_d), m_m(m.get()), m_wrt(m_ed.m_ctrl), m_params(params),  m_func(mj_step){
            m_wrt = select_ptr(m_params.m_wrt_id);
            m_func = select_mode(m_params.m_mode_id);
            m_sens_res = Eigen::MatrixXd(m_m->nsensordata, m_wrt.size());
            if (m_params.m_mode_id == Mode::Fwd)
                m_func_res = Eigen::MatrixXd(m_m->nq + m_m->nv, m_wrt.size());
            else
                m_func_res = Eigen::MatrixXd(m_m->nv, m_wrt.size());
        };


        ~MjDerivative(){
            mj_deleteData(m_d);
        };


        const Eigen::MatrixXd &inv(const MjDataVecView &ed){
            mjcb_control = [](const mjModel* m, mjData* d){};
            copy_data(m_m, ed.m_d, m_ed.m_d);
            mj_inverse(m_m, m_ed.m_d);
            for (int i = 0; i < m_wrt.size(); ++i) {
                perturb(i);
                m_func(m_m, m_ed.m_d);
                m_func_res.block(0, i, m_m->nv, 1) = (m_ed.m_qfrc_inverse - ed.m_qfrc_inverse) / m_params.m_eps;
                copy_data(m_m, ed.m_d, m_ed.m_d);
                mj_inverse(m_m, m_ed.m_d);
            }
            return m_func_res;
        };


        const Eigen::MatrixXd &fwd(const MjDataVecView &ed){
            mjcb_control = [](const mjModel* m, mjData* d){};
            copy_data(m_m, ed.m_d, m_ed.m_d);
            mj_forward(m_m, m_ed.m_d);
            for (int i = 0; i < m_wrt.size(); ++i) {
                perturb(i);
                m_func(m_m, m_ed.m_d);
                m_func_res.block(0, i, m_m->nq, 1) = (m_ed.m_pos - ed.m_pos) / m_params.m_eps;
                m_func_res.block(m_m->nq, i, m_m->nv, 1) = (m_ed.m_vel - ed.m_vel) / m_params.m_eps;
                copy_data(m_m, ed.m_d, m_ed.m_d);
                mj_forward(m_m, m_ed.m_d);
            }
            return m_func_res;
        };


        const Eigen::MatrixXd &func(const MjDataVecView &ed){
            if(m_params.m_mode_id == Mode::Fwd)
                return fwd(ed);
            else
                return inv(ed);
        };


        const Eigen::MatrixXd &sensors(const MjDataVecView &ed) {
            mjcb_control = [](const mjModel* m, mjData* d){};
            copy_data(m_m, ed.m_d, m_ed.m_d);
            for (int i = 0; i < m_wrt.size(); ++i){
                perturb(i);
                m_func(m_m, m_ed.m_d);
                m_sens_res.col(i) = (m_ed.m_sens - ed.m_sens) / m_params.m_eps;
                copy_data(m_m, ed.m_d, m_ed.m_d);
            }
            return m_sens_res;
        };

    private:
        // Deal with free and ball joints
        void perturb(const int idx) {
            if (m_params.m_wrt_id == Wrt::Pos) {
                // get quaternion address if applicable
                const auto jid = m_m->dof_jntid[idx];
                int quatadr = -1, dofpos = 0;
                if (m_m->jnt_type[jid] == mjJNT_BALL) {
                    quatadr = m_m->jnt_qposadr[jid];
                    dofpos = idx - m_m->jnt_dofadr[jid];
                } else if (m_m->jnt_type[jid] == mjJNT_FREE && idx >= m_m->jnt_dofadr[jid] + 3) {
                    quatadr = m_m->jnt_qposadr[jid] + 3;
                    dofpos = idx - m_m->jnt_dofadr[jid] - 3;
                }

                // apply quaternion or simple perturbation
                if (quatadr >= 0) {
                    mjtNum angvel[3] = {0, 0, 0};
                    angvel[dofpos] = m_params.m_eps;
                    mju_quatIntegrate(m_wrt.data() + quatadr, angvel, 1);
                } else
                    m_wrt.data()[m_m->jnt_qposadr[jid] + idx - m_m->jnt_dofadr[jid]] += m_params.m_eps;
            } else {
                m_wrt(idx) = m_wrt(idx) + m_params.m_eps;
            }
        }

        std::reference_wrapper<EigenMatrixXMap> select_ptr(const Wrt wrt) {
            switch (wrt) {
                case Wrt::Ctrl:
                    return m_ed.m_ctrl;
                    break;
                case Wrt::Pos:
                    return m_ed.m_pos;
                    break;
                case Wrt::Vel:
                    return m_ed.m_vel;
                    break;
                case Wrt::Acc:
                    return m_ed.m_acc;
                    break;
                default:
                    return m_ed.m_ctrl;
            }
        }

        mjfGeneric select_mode(const Mode mode){
            switch (mode) {
                case Mode::Fwd:
                    return mj_step;
                    break;
                case Mode::Inv:
                    return mj_inverse;
                    break;
                default:
                    return {};
            }
        }

    private:
        mjData* m_d;
        MjDataVecView m_ed;
        const mjModel *m_m;
        Eigen::MatrixXd m_func_res;
        Eigen::MatrixXd m_sens_res;
        EigenMatrixXMap& m_wrt;
        const MjDerivativeParams m_params;
        mjfGeneric m_func;
    };

    PYBIND11_MODULE(_derivative, pymodule) {
    namespace py = ::pybind11;

    py::enum_<Wrt>(pymodule, "Wrt", py::arithmetic())
    .value("Pos", Wrt::Pos)
    .value("Vel", Wrt::Vel)
    .value("Acc", Wrt::Acc)
    .value("Ctrl", Wrt::Ctrl)
    .export_values();

    py::enum_<Mode>(pymodule, "Mode", py::arithmetic())
    .value("Inv", Mode::Inv)
    .value("Fwd", Mode::Fwd)
    .export_values();

    py::class_<MjDerivativeParams>(pymodule,"MjDerivativeParams")
    .def (py::init<double, const Wrt, const Mode>())
    .def_readwrite("eps", &MjDerivativeParams::m_eps);

    py::class_<MjDerivative>(pymodule,"MjDerivative")
    .def (py::init<const MjModelWrapper &, const MjDerivativeParams>())
    .def("wrt_dynamics", &MjDerivative::func)
    .def("wrt_sensors", &MjDerivative::sensors);

    py::class_<MjDataVecView>(pymodule,"MjDataVecView")
    .def (py::init<const MjModelWrapper &, MjDataWrapper &>())
    .def (py::init<const raw::MjModel *, raw::MjData *>());
    }
}
}
