/*
 *  multiecho.cpp
 *
 *  Created by Tobias Wood on 27/01/2015.
 *  Copyright (c) 2015 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NumericalDiff>

#include "QI/Types.h"
#include "QI/Util.h"
#include "QI/Option.h"
#include "QI/Sequences/SpinEcho.h"
#include "Filters/ApplyAlgorithmFilter.h"
#include "Filters/ReorderImageFilter.h"
#include "itkTimeProbe.h"
#include "itkImageFileReader.h"

using namespace std;
using namespace Eigen;

/*
 * Base class for the 3 different algorithms
 */
class RelaxAlgo : public Algorithm<double> {
private:
    const shared_ptr<QI::SCD> m_model = make_shared<QI::SCD>();
protected:
    shared_ptr<QI::MultiEcho> m_sequence;
    double m_clampLo = -numeric_limits<double>::infinity();
    double m_clampHi = numeric_limits<double>::infinity();
    double m_thresh = -numeric_limits<double>::infinity();

    void clamp_and_treshold(const TInput &data, TArray &outputs, TArray &resids,
                            const double PD, const double T2) const {
        if (PD > m_thresh) {
            outputs[0] = PD;
            outputs[1] = QI::clamp(T2, m_clampLo, m_clampHi);
            ArrayXcd theory = QI::One_MultiEcho(m_sequence->TE(), m_sequence->TR(), PD, 0., T2); // T1 isn't modelled, set to 0 for instant recovery
            resids = data.array() - theory.abs();
        } else {
            outputs.setZero();
            resids.setZero();
        }
    }

public:
    void setSequence(shared_ptr<QI::MultiEcho> &s) { m_sequence = s; }
    void setClamp(double lo, double hi) { m_clampLo = lo; m_clampHi = hi; }
    void setThresh(double t) { m_thresh = t; }
    size_t numInputs() const override { return m_sequence->count(); }
    size_t numConsts() const override { return 1; }
    size_t numOutputs() const override { return 2; }
    size_t dataSize() const override { return m_sequence->size(); }

    virtual TArray defaultConsts() override {
        // B1
        TArray def = TArray::Ones(2);
        return def;
    }
};

class LogLinAlgo: public RelaxAlgo {
public:
    virtual void apply(const TInput &data, const TArray &inputs,
                       TArray &outputs, TArray &resids, TIterations &its) const override
    {
            // Set up echo times array
        MatrixXd X(m_sequence->size(), 2);
        X.col(0) = m_sequence->m_TE;
        X.col(1).setOnes();
        VectorXd Y = data.array().log();
        VectorXd b = (X.transpose() * X).partialPivLu().solve(X.transpose() * Y);
        double PD = exp(b[1]);
        double T2 = -1 / b[0];
        clamp_and_treshold(data, outputs, resids, PD, T2);
        its = 1;
    }
};

class ARLOAlgo : public RelaxAlgo {
public:
    virtual void apply(const TInput &data, const TArray &inputs,
                       TArray &outputs, TArray &resids, TIterations &its) const override
    {
        const double dTE_3 = (m_sequence->m_ESP / 3);
        double si2sum = 0, sidisum = 0;
        for (int i = 0; i < m_sequence->size() - 2; i++) {
            const double si = dTE_3 * (data(i) + 4*data(i+1) + data(i+2));
            const double di = data(i) - data(i+2);
            si2sum += si*si;
            sidisum += si*di;
        }
        double T2 = (si2sum + dTE_3*sidisum) / (dTE_3*si2sum + sidisum);
        double PD = (data.array() / exp(-m_sequence->m_TE / T2)).mean();
        clamp_and_treshold(data, outputs, resids, PD, T2);
        its = 1;
    }
};

class RelaxFunctor : public DenseFunctor<double> {
    protected:
        const shared_ptr<QI::SequenceBase> m_sequence;
        const ArrayXd m_data;
        const shared_ptr<QI::SCD> m_model = make_shared<QI::SCD>();

    public:
        RelaxFunctor(shared_ptr<QI::SequenceBase> cs, const ArrayXd &data) :
            DenseFunctor<double>(2, cs->size()),
            m_sequence(cs), m_data(data)
        {
            assert(static_cast<size_t>(m_data.rows()) == values());
        }

        int operator()(const Ref<VectorXd> &params, Ref<ArrayXd> diffs) const {
            eigen_assert(diffs.size() == values());
            VectorXd fullp(5);
            fullp << params(0), 0, params(1), 0, 1.0; // Fix B1 to 1.0 for now
            ArrayXcd s = m_sequence->signal(m_model, fullp);
            diffs = s.abs() - m_data;
            return 0;
        }
};

class NonLinAlgo : public RelaxAlgo {
private:
    size_t m_iterations = 5;
public:
    void setIterations(size_t n) { m_iterations = n; }

    virtual void apply(const TInput &data, const TArray &inputs,
                       TArray &outputs, TArray &resids, TIterations &its) const override
    {
        RelaxFunctor f(m_sequence, data);
        NumericalDiff<RelaxFunctor> nDiff(f);
        LevenbergMarquardt<NumericalDiff<RelaxFunctor>> lm(nDiff);
        lm.setMaxfev(m_iterations * (m_sequence->size() + 1));
        // Just PD & T2 for now
        // Basic guess of T2=50ms
        VectorXd p(2); p << data(0), 0.05;
        lm.minimize(p);
        clamp_and_treshold(data, outputs, resids, p[0], p[1]);
        its = lm.iterations();
    }
};

//******************************************************************************
// Main
//******************************************************************************
int main(int argc, char **argv) {
    Eigen::initParallel();
    QI::OptionList opts("Usage is: qimultiecho [options] input_file");
    QI::Switch all_residuals('r',"resids","Write out per flip-angle residuals", opts);
    QI::Option<int> num_threads(4,'T',"threads","Use N threads (default=4, 0=hardware limit)", opts);
    QI::Option<int> its(15,'i',"its","Max iterations for NLLS (default 15)", opts);
    QI::Switch reorder('R',"reorder","Re-order data", opts);
    QI::Option<float> clamp(std::numeric_limits<float>::infinity(),'c',"clamp","Clamp tau between 0 and value", opts);
    QI::ImageOption<QI::VolumeF> mask('m', "mask", "Mask input with specified file", opts);
    QI::ImageOption<QI::VolumeF> B1('b', "B1", "B1 Map file (ratio)", opts);
    QI::Switch flex('f',"flex","Use flexible TEs", opts);
    QI::EnumOption algorithm("lan",'l','a',"algo","Choose algorithm (l/a/n)", opts);
    QI::Option<std::string> outPrefix("", 'o', "out","Add a prefix to output filenames", opts);
    QI::Switch suppress('n',"no-prompt","Suppress input prompts", opts);
    QI::Switch verbose('v',"verbose","Print more information", opts);
    QI::Help help(opts);
    std::vector<std::string> nonopts = opts.parse(argc, argv);
    if (nonopts.size() != 1) {
        std::cerr << opts << std::endl;
        std::cerr << "No input filename specified." << std::endl;
        return EXIT_FAILURE;
    }
    itk::MultiThreader::SetGlobalMaximumNumberOfThreads(*num_threads);
    shared_ptr<RelaxAlgo> algo;
    switch (*algorithm) {
        case 'l': algo = make_shared<LogLinAlgo>(); if (*verbose) cout << "LogLin algorithm selected." << endl; break;
        case 'a': algo = make_shared<ARLOAlgo>(); if (*verbose) cout << "ARLO algorithm selected." << endl; break;
        case 'n': algo = make_shared<NonLinAlgo>(); if (*verbose) cout << "Non-linear algorithm (Levenberg Marquardt) selected." << endl; break;
    }
    shared_ptr<QI::MultiEcho> multiecho;
    if (*flex)
        multiecho = make_shared<QI::MultiEchoFlex>(cin, !*suppress);
    else
        multiecho = make_shared<QI::MultiEcho>(cin, !*suppress);
    algo->setSequence(multiecho);
    auto apply = itk::ApplyAlgorithmFilter<RelaxAlgo>::New();
    apply->SetAlgorithm(algo);
    apply->SetMask(*mask);
    if (*verbose) cout << "Opening input file: " << nonopts[0] << endl;
    auto inputFile = itk::ImageFileReader<QI::SeriesF>::New();
    inputFile->SetFileName(nonopts[0]);
    inputFile->Update(); // Need to know the length of the vector for re-ordering
    size_t nVols = inputFile->GetOutput()->GetLargestPossibleRegion().GetSize()[3] / multiecho->size();
    auto inputData = QI::ReorderSeriesF::New();
    inputData->SetInput(inputFile->GetOutput());
    if (*reorder)
        inputData->SetStride(nVols);
    inputData->Update();

    auto PDoutput = itk::TileImageFilter<QI::VolumeF, QI::SeriesF>::New();
    auto T2output = itk::TileImageFilter<QI::VolumeF, QI::SeriesF>::New();
    itk::FixedArray<unsigned int, 4> layout;
    layout[0] = layout[1] = layout[2] = 1; layout[3] = nVols;
    PDoutput->SetLayout(layout);
    T2output->SetLayout(layout);
    if (*verbose) cout << "Processing" << endl;
    auto inputVector = QI::SeriesToVectorF::New();
    inputVector->SetInput(inputData->GetOutput());
    inputVector->SetBlockSize(multiecho->size());
    vector<QI::VolumeF::Pointer> PDimgs(nVols), T2imgs(nVols);
    for (size_t i = 0; i < nVols; i++) {
        inputVector->SetBlockStart(i * multiecho->size());

        apply->SetAlgorithm(algo);
        apply->SetInput(0, inputVector->GetOutput());
        apply->Update();

        PDimgs.at(i) = apply->GetOutput(0);
        T2imgs.at(i) = apply->GetOutput(1);
        PDimgs.at(i)->DisconnectPipeline();
        T2imgs.at(i)->DisconnectPipeline();

        PDoutput->SetInput(i, PDimgs.at(i));
        T2output->SetInput(i, T2imgs.at(i));
    }
    if (*verbose) cout << "Writing output with prefix: " << *outPrefix << endl;
    PDoutput->UpdateLargestPossibleRegion();
    T2output->UpdateLargestPossibleRegion();
    *outPrefix = *outPrefix + "ME_";
    QI::WriteImage(PDoutput->GetOutput(), *outPrefix + "M0" + QI::OutExt());
    QI::WriteImage(T2output->GetOutput(), *outPrefix + "T" + QI::OutExt());
    //QI::writeResiduals(apply->GetResidOutput(), outPrefix, all_residuals);

    return EXIT_SUCCESS;
}

