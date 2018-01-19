/*
 *  SteadyStateSequence.h
 *
 *  Copyright (c) 2016 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#ifndef SEQUENCES_STEADYSTATE_H
#define SEQUENCES_STEADYSTATE_H

#include "SequenceBase.h"

namespace QI {

class SteadyState : public SequenceBase {
    public:
        SteadyState();
        SteadyState(const Eigen::ArrayXd &flip, const double TR);

        size_t size() const override { return m_flip.rows(); }
};

class SPGRSimple : public SteadyState {
    public:
        SPGRSimple(const Eigen::ArrayXd &flip, const double TR);
        SPGRSimple(std::istream &istr, const bool prompt);
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream &os) const override;
        std::string name() const override { return "SPGR"; }
        Eigen::ArrayXd weights(const double f0 = 0.0) const override;
};

class SPGREcho : public SPGRSimple {
public:
    double m_TE;
    SPGREcho(const Eigen::ArrayXd &flip, const double TR, const double TE);
    SPGREcho(std::istream &istr, const bool prompt);
    Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
    void write(std::ostream &os) const override;
    std::string name() const override { return "SPGR_Echo"; }
};

class SPGRFinite : public SPGRSimple {
    public:
        double m_Trf, m_TE;
        SPGRFinite(const Eigen::ArrayXd &flip, const double TR, const double Trf, const double TE);
        SPGRFinite(std::istream &istr, const bool prompt);
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream &os) const override;
        std::string name() const override { return "SPGR_Finite"; }
};

class SSFPSimple : public SteadyState {
    protected:
        Eigen::ArrayXd m_phi;               // These store just 1 set of flips/phase-incs
        Eigen::ArrayXd m_allFlip, m_allPhi; // These store the full set of flips/phase-incs
        void setupAll();

    public:
        SSFPSimple();
        SSFPSimple(const Eigen::ArrayXd &flip, const double TR, const Eigen::ArrayXd &phases);
        SSFPSimple(std::istream &istr, const bool prompt);
        size_t size() const override { return m_allFlip.rows(); }
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream& os) const override;
        std::string name() const override { return "SSFP"; }
        virtual const Eigen::ArrayXd &phase_incs() const { return m_phi; }
        const Eigen::ArrayXd &allFlip() const { return m_allFlip; }
        const Eigen::ArrayXd &allPhi() const { return m_allPhi; }
        Eigen::ArrayXd weights(const double f0) const override;
};

class SSFPEcho : public SSFPSimple {
public:
    SSFPEcho();
    SSFPEcho(std::istream &istr, const bool prompt);

    Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
    Eigen::ArrayXd  signal_magnitude(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
    std::string name() const override { return "SSFPEcho"; }
};

class SSFPEchoFlex : public SSFPEcho {
public:
    SSFPEchoFlex(std::istream &istr, const bool prompt);
    std::string name() const override { return "SSFPEchoFlex"; }
    const Eigen::ArrayXd &flip() const override { return m_allFlip; }
    const Eigen::ArrayXd &phase_incs() const override { return m_allPhi; }
};

class SSFPFinite : public SSFPSimple {
    public:
        double m_Trf;
        SSFPFinite(const Eigen::ArrayXd &flip, const double TR, const double Trf, const Eigen::ArrayXd &phases);
        SSFPFinite(std::istream &istr, const bool prompt);
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream& os) const override;
        std::string name() const override { return "SSFP_Finite"; }
};

class SSFP_GS : public SteadyState {
    public:
        SSFP_GS(std::istream &istr, const bool prompt);
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream& os) const override;
        std::string name() const override { return "SSFP_Ellipse"; }
};

class AFI : public SteadyState {
protected:
    double m_TR1, m_TR2;
    public:
        AFI(std::istream &istr, const bool prompt);

        size_t size() const override { return 2; }
        Eigen::ArrayXcd signal(std::shared_ptr<Model> m, const Eigen::VectorXd &par) const override;
        void write(std::ostream& os) const override;
        std::string name() const override { return "AFI"; }
};

} // End namespace QI

#endif // SEQUENCES_STEADYSTATE_H