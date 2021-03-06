/*
 *  Spline.cpp
 *
 *  Copyright (c) 2017 Tobias Wood.
 *
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this
 *  file, you can obtain one at http://mozilla.org/MPL/2.0/.
 *
 */

#include "Spline.h"
#include "Macro.h"
#include <iostream>
#include <iomanip>

namespace QI {

SplineInterpolator::SplineInterpolator() :
    m_min(0.), m_width(0.)
{ }

// Sort input in ascending order



SplineInterpolator::SplineInterpolator(Eigen::ArrayXd const &x, Eigen::ArrayXd const &y) {
    if (x.size() != y.size()) {
        QI_FAIL("Input vectors to spline must be same size");
    }
    std::vector<std::size_t> indices(x.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j){return x[i] < x[j];});

    Eigen::ArrayXd sx(x.size()), sy(y.size());
    std::transform(indices.begin(), indices.end(), &sx[0], [&](std::size_t i){return x[i]; });
    std::transform(indices.begin(), indices.end(), &sy[0], [&](std::size_t i){return y[i]; });

    m_min = sx[0];
    m_width = sx[sx.size()-1] - m_min;
    const Eigen::ArrayXd scaledx = (sx - m_min) / m_width;
    m_spline = Eigen::SplineFitting<TSpline>::Interpolate(sy.transpose(), std::min<int>(x.rows() - 1, 3), scaledx.transpose());
}

double SplineInterpolator::operator()(const double &x) const {
    const double sx = scale(x);
    const double val = m_spline(sx)[0];
    return val;
}

double SplineInterpolator::scale(const double &x) const {
    return (x - m_min) / m_width;
}

} // End namespace QI