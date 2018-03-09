/*
 *  qi_ase_oef.cpp
 *
 *  Copyright (c) 2018 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include <iostream>
#include <Eigen/Dense>

#include "Util.h"
#include "ImageIO.h"
#include "Args.h"
#include "ApplyTypes.h"
#include "MultiEchoSequence.h"
#include "SequenceCereal.h"

class ASEAlgo : public QI::ApplyF::Algorithm {
protected:
    const QI::MultiEchoSequence m_sequence;
    const int m_inputsize;
    int m_linear_count;
    const double m_B0;

    // Constants for calculations
    const double kappa = 0.03; // Conversion factor
    const double gamma = 42.577e6; // Gyromagnetic Ratio
    const double delta_X0 = 0.264e-6; // Difference in susceptibility of oxy and fully de-oxy blood
    const double Hb = 0.34 / kappa; // Hct = 0.34;

public:
    ASEAlgo(const QI::MultiEchoSequence& seq, const int inputsize, const double B0) :
        m_sequence(seq), m_B0(B0), m_inputsize(inputsize)
    {
        // 2.94 is dHb for an OEF of 26%
        const double Tc = 2.0 / ((4./3.) * M_PI * gamma * B0 * delta_X0 * 0.75);
        m_linear_count = (m_sequence.TE > Tc).count();
        QI_DB( B0 );
        QI_DB( Tc );
        QI_DBVEC( m_sequence.TE );
        QI_DBVEC( (m_sequence.TE > Tc) );
        QI_DB( m_linear_count );
    }

    size_t numInputs() const override  { return 1; }
    size_t numConsts() const override  { return 0; }
    size_t numOutputs() const override { return 4; }
    size_t dataSize() const override   { return m_inputsize; }
    size_t outputSize() const override { return 1; }
    TOutput zero() const override {
        TOutput z;
        return z;
    }

    std::vector<float> defaultConsts() const override {
        std::vector<float> def;
        return def;
    }

    const std::vector<std::string> &names() const {
        static std::vector<std::string> _names = {"R2prime", "DBV", "OEF", "dHb"};
        return _names;
    }

    bool apply(const std::vector<TInput> &inputs, const std::vector<TConst> &consts,
               const TIndex &index, // Unused
               std::vector<TOutput> &outputs, TOutput &residual,
               TInput &resids, TIterations &its) const override
    {
        const Eigen::Map<const Eigen::ArrayXXf> input(inputs[0].GetDataPointer(), m_sequence.size(), inputs[0].Size() / m_sequence.size());
        Eigen::ArrayXd data = input.cast<double>().rowwise().mean();

        // Eigen::MatrixXd X(m_linear_count, 2);
        // X.col(0) = m_sequence.TE.tail(m_linear_count);
        // X.col(1).setOnes();
        // Eigen::VectorXd Y = data.tail(m_linear_count).log();
        // Eigen::VectorXd b = (X.transpose() * X).partialPivLu().solve(X.transpose() * Y);
        // const double R2prime = -b[0];
        // const double logS0_linear = b[1];

        Eigen::ArrayXd linear_data = data.tail(m_linear_count);
        const double dTE_3 = (m_sequence.ESP / 3);
        double si2sum = 0, sidisum = 0;
        for (int i = 0; i < m_linear_count - 2; i++) {
            const double si = dTE_3 * (linear_data(i) + 4*linear_data(i+1) + linear_data(i+2));
            const double di = linear_data(i) - linear_data(i+2);
            si2sum += si*si;
            sidisum += si*di;
        }
        double R2prime = 1 / ((si2sum + dTE_3*sidisum) / (dTE_3*si2sum + sidisum));
        double S0_linear = (linear_data.array() / exp(-m_sequence.TE.tail(m_linear_count) * R2prime)).mean();

        const double DBV = log(S0_linear) - log(data[0]);

        const double dHb = 3*R2prime / (DBV * 4 * gamma * M_PI * delta_X0 * kappa * m_B0);
        const double OEF = dHb / Hb;

        // QI_DB( input );
        // QI_DBVEC( data );
        // QI_DB( R2prime );
        // QI_DB( log(S0_linear) );
        // QI_DB( log(data[0]) );
        // QI_DB( DBV );
        // QI_DB( dHb );
        // QI_DB( OEF );

        outputs[0] = R2prime;
        outputs[1] = DBV*100;
        outputs[2] = OEF*100;
        outputs[3] = dHb;
        residual = 0;
        resids.Fill(0.);
        its = 0;
        return true;
    }
};

/*
 * Main
 */
int main(int argc, char **argv) {
    Eigen::initParallel();
    args::ArgumentParser parser("Calculates the OEF from ASE data.\nhttp://github.com/spinicist/QUIT");
    args::Positional<std::string> input_path(parser, "ASE_FILE", "Input ASE file");
    args::HelpFlag help(parser, "HELP", "Show this help message", {'h', "help"});
    args::Flag     verbose(parser, "VERBOSE", "Print more information", {'v', "verbose"});
    args::ValueFlag<int> threads(parser, "THREADS", "Use N threads (default=4, 0=hardware limit)", {'T', "threads"}, 4);
    args::ValueFlag<std::string> outarg(parser, "OUTPREFIX", "Add a prefix to output filename", {'o', "out"});
    args::ValueFlag<std::string> mask(parser, "MASK", "Only process voxels within the mask", {'m', "mask"});
    args::ValueFlag<double> B0(parser, "B0", "Field-strength (Tesla), default 3", {'B', "B0"}, 3.0);
    args::ValueFlag<std::string> subregion(parser, "SUBREGION", "Process subregion starting at voxel I,J,K with size SI,SJ,SK", {'s', "subregion"});
    QI::ParseArgs(parser, argc, argv);
    if (verbose) std::cout << "Starting " << argv[0] << std::endl;
    if (verbose) std::cout << "Reading ASE data from: " << QI::CheckPos(input_path) << std::endl;
    auto input = QI::ReadVectorImage(QI::CheckPos(input_path));

    auto sequence = QI::ReadSequence<QI::MultiEchoSequence>(std::cin, verbose);
    std::shared_ptr<ASEAlgo> algo = std::make_shared<ASEAlgo>(sequence, input->GetNumberOfComponentsPerPixel(), B0.Get());
    auto apply = QI::ApplyF::New();
    apply->SetVerbose(verbose);
    apply->SetAlgorithm(algo);
    apply->SetOutputAllResiduals(false);
    if (verbose) std::cout << "Using " << threads.Get() << " threads" << std::endl;
    apply->SetPoolsize(threads.Get());
    apply->SetSplitsPerThread(threads.Get());
    apply->SetInput(0, input);
    if (mask) apply->SetMask(QI::ReadImage(mask.Get()));
    if (subregion) {
        apply->SetSubregion(QI::RegionArg(args::get(subregion)));
    }
    if (verbose) {
        std::cout << "Processing" << std::endl;
        auto monitor = QI::GenericMonitor::New();
        apply->AddObserver(itk::ProgressEvent(), monitor);
    }
    apply->Update();
    if (verbose) {
        std::cout << "Elapsed time was " << apply->GetTotalTime() << "s" << std::endl;
    }
    const std::string outPrefix = outarg ? outarg.Get() : QI::Basename(input_path.Get());
    for (size_t i = 0; i < algo->numOutputs(); i++) {
        const std::string fname = outPrefix + "_" + algo->names()[i] + QI::OutExt();
        std::cout << "Writing file: " << fname << std::endl;
        QI::WriteImage(apply->GetOutput(i), fname);
    }
    return EXIT_SUCCESS;
}
