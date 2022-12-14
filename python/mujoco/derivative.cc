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
        State = 0, Ctrl = 1
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
        explicit MjDerivative(const MjModelWrapper &m, MjDataWrapper &d, const MjDerivativeParams& params) :
                m_d(mj_makeData(m.get())), m_ed_internal(m.get(), m_d), m_ed_external(m, d), m_m(m.get()), m_wrts({m_ed_internal.m_ctrl}), m_params(params), m_func(mj_step){
            m_wrts = select_ptr(m_params.m_wrt_id);
            m_func = select_mode(m_params.m_mode_id);
            auto cols = 0; for (const EigenMatrixXMap& wrt: m_wrts){cols += wrt.size();}
            m_sens_res = Eigen::MatrixXd(m_m->nsensordata, cols);
            if (m_params.m_mode_id == Mode::Fwd)
                m_func_res = Eigen::MatrixXd(m_m->nq + m_m->nv, cols);
            else
                m_func_res = Eigen::MatrixXd(m_m->nv, cols);
        };


        ~MjDerivative(){
            mj_deleteData(m_d);
        };


        const Eigen::MatrixXd &inv(){
            mjcb_control = [](const mjModel* m, mjData* d){};
            long col = 0;
            for(EigenMatrixXMap& wrt: m_wrts)
            {
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
                for (int i = 0; i < wrt.size(); ++i) {
                    perturb(i, wrt);
                    m_func(m_m, m_ed_internal.m_d);
                    m_func_res.block(0, col + i, m_m->nv, 1) = m_ed_internal.m_qfrc_inverse;
                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    m_func(m_m, m_ed_internal.m_d);
                    m_func_res.block(0, col + i, m_m->nv, 1) -= m_ed_internal.m_qfrc_inverse;
                    m_func_res.block(0, col + i, m_m->nv, 1) /= m_params.m_eps;
                }
                col += wrt.size();
            }

            return m_func_res;
        };


        const Eigen::MatrixXd &fwd(){
            mjcb_control = [](const mjModel* m, mjData* d){};
            long col = 0;
            for(EigenMatrixXMap& wrt: m_wrts)
            {
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
                for (int i = 0; i < wrt.size(); ++i) {
                    perturb(i, wrt);
                    m_func(m_m, m_ed_internal.m_d);
                    m_func_res.block(0, col + i, m_m->nq, 1) = m_ed_internal.m_pos;
                    m_func_res.block(m_m->nq, col + i, m_m->nv, 1) = m_ed_internal.m_vel;
                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    // f(u + e) - f(u) / eps
                    m_func(m_m, m_ed_internal.m_d);
                    m_func_res.block(0, col + i, m_m->nq, 1) -= m_ed_internal.m_pos;
                    m_func_res.block(m_m->nq, col + i, m_m->nv, 1) -= m_ed_internal.m_vel;
                    m_func_res.block(0, col + i, m_m->nq, 1) /= m_params.m_eps;
                    m_func_res.block(m_m->nq, col + i, m_m->nv, 1) /= m_params.m_eps;
                }
                col += wrt.size();
            }

            return m_func_res;
        };


        const Eigen::MatrixXd &output(){
            if(m_params.m_mode_id == Mode::Fwd)
                return fwd();
            else
                return inv();
        };


        const Eigen::MatrixXd &sensor() {
            mjcb_control = [](const mjModel* m, mjData* d){};
            long col = 0;
            for(EigenMatrixXMap& wrt: m_wrts)
            {
                copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);
                for (int i = 0; i < wrt.size(); ++i) {
                    perturb(i, wrt);
                    m_func(m_m, m_ed_internal.m_d);
                    m_sens_res.col(i + col) = m_ed_internal.m_sens;
                    copy_data(m_m, m_ed_external.m_d, m_ed_internal.m_d);

                    m_func(m_m, m_ed_internal.m_d);
                    m_sens_res.col(i + col) -= m_ed_internal.m_sens;
                    m_sens_res.col(i + col) /= m_params.m_eps;
                }
                col += wrt.size();
            }
            return m_sens_res;
        };


    private:
        // Deal with free and ball joints
        void perturb(const int idx, EigenMatrixXMap& wrt) {
            if (&wrt == &m_ed_internal.m_pos) {
                // get quaternion address if applicable`
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
                    mju_quatIntegrate(wrt.data() + quatadr, angvel, 1);
                } else
                    wrt.data()[m_m->jnt_qposadr[jid] + idx - m_m->jnt_dofadr[jid]] += m_params.m_eps;
            } else {
                wrt(idx) = wrt(idx) + m_params.m_eps;
            }
        }


        std::vector<std::reference_wrapper<EigenMatrixXMap>> select_ptr(const Wrt wrt) {
            if(m_params.m_mode_id == Mode::Fwd)
                switch (wrt) {
                    case Wrt::Ctrl:
                        return {m_ed_internal.m_ctrl};
                        break;
                    case Wrt::State:
                        return{m_ed_internal.m_pos, m_ed_internal.m_vel};
                    default:
                        return {m_ed_internal.m_ctrl};
                }
            else
                return{m_ed_internal.m_pos, m_ed_internal.m_vel, m_ed_internal.m_acc};
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
                    return mj_step;
            }
        }


    private:
        mjData* m_d;
        MjDataVecView m_ed_internal;
        const MjDataVecView m_ed_external;
        const mjModel *m_m;
        Eigen::MatrixXd m_func_res;
        Eigen::MatrixXd m_sens_res;
        std::vector<std::reference_wrapper<EigenMatrixXMap>> m_wrts;
        const MjDerivativeParams m_params;
        mjfGeneric m_func;
    };

    PYBIND11_MODULE(_derivative, pymodule) {
    namespace py = ::pybind11;

    py::enum_<Wrt>(pymodule, "Wrt", py::arithmetic())
    .value("State", Wrt::State)
    .value("Ctrl", Wrt::Ctrl)
    .export_values();

    py::enum_<Mode>(pymodule, "Mode", py::arithmetic())
    .value("Inv", Mode::Inv)
    .value("Fwd", Mode::Fwd)
    .export_values();

    py::class_<MjDerivativeParams>(pymodule,"MjDerivativeParams")
    .def (py::init<double, const Wrt, const Mode>());

    py::class_<MjDerivative>(pymodule,"MjDerivative")
    .def (py::init<const MjModelWrapper &, MjDataWrapper &, const MjDerivativeParams>())
    .def("func", &MjDerivative::output)
    .def("sensors", &MjDerivative::sensor);

    py::class_<MjDataVecView>(pymodule,"MjDataVecView")
    .def (py::init<const MjModelWrapper &, MjDataWrapper &>())
    .def (py::init<const raw::MjModel *, raw::MjData *>());
    }
}
}
