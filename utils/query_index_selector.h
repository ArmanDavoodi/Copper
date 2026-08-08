#ifndef QUERY_INDEX_SELECTOR_H_
#define QUERY_INDEX_SELECTOR_H_

#include <iostream>
#include <random>
#include <cmath>
#include <numbers>
#include <algorithm>

#include "debug.h"

namespace divftree {

class IndexSelector {
private:
    // Fixed min value per your specification
    static constexpr size_t MIN_BOUND = 0;

    // Set exactly once at application startup
    size_t max_bound_ = 9999; 
    double mean_val_ = 5000.0;
    double stddev_ = 0.0;

    // Cached constants computed once at startup
    double cdf_min_ = 0.0;
    double cdf_max_ = 1.0;
    bool use_uniform_ = false;
    bool use_mean_only_ = false;

    // Standard Normal CDF
    double normal_cdf(double x) const {
        return 0.5 * (1.0 + std::erf((x - mean_val_) / (stddev_ * std::numbers::sqrt2)));
    }

    // High-accuracy Inverse CDF (Acklam's Approximation)
    double normal_inverse_cdf(double p) const {
        static constexpr double a[] = {
            -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479800018616e+01,  2.506628277459239e+00
        };
        static constexpr double b[] = {
            -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01
        };
        static constexpr double c[] = {
            -2.787189311384079e-01, -2.277938898656075e+00,  4.071618331404194e+00,
            -4.685391282270586e+00,  2.439889753832790e+00, -5.351475530854471e-01
        };
        static constexpr double d[] = {
             7.646752923027944e-03,  1.595271138250280e-01,  5.738377443405415e-01,
             3.100181992111739e-01
        };

        double q, r;
        if (p < 0.02425) { // Lower tail
            q = std::sqrt(-2.0 * std::log(p));
            // Horner's method evaluated cleanly via array indices
            r = (((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) /
                (((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4]) * q + 1.0);
        } else if (p <= 0.97575) { // Central region
            q = p - 0.5;
            r = q * q;
            r = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) /
                (((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0) * q);
        } else { // Upper tail
            q = std::sqrt(-2.0 * std::log(1.0 - p));
            r = -(((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) /
                (((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4]) * q + 1.0);
        }
        
        return mean_val_ + r * stddev_;
    }

public:
    // Call this immediately after parsing your setup files
    void Configure(size_t total_num_queries, double stddev) {
        // Adjusting parsed semi-closed bound (e.g., 10000 -> 9999 inclusive)
        max_bound_ = total_num_queries - 1; 
        stddev_ = stddev;

        // Mean is always half of our active top bound since min is 0
        mean_val_ = static_cast<double>(max_bound_) / 2.0;

        use_mean_only_ = (stddev_ == 0.0);
        use_uniform_ = (stddev_ < 0.0 || stddev_ > static_cast<double>(max_bound_) * 2.0);

        if (!use_mean_only_ && !use_uniform_) {
            // Compute the fixed CDF scaling limits once
            cdf_min_ = normal_cdf(static_cast<double>(MIN_BOUND) - 0.5);
            cdf_max_ = normal_cdf(static_cast<double>(max_bound_) + 0.5);
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "IndexSelector configured: min=%zu, max=%zu, mean=%.2f, stddev=%.2f, use_uniform=%d, use_mean_only=%d",
                MIN_BOUND, max_bound_, mean_val_, stddev_, use_uniform_, use_mean_only_);
    }

    // High-performance hot path
    size_t Select() const {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());

        if (use_mean_only_) return static_cast<size_t>(std::round(mean_val_));

        if (use_uniform_) {
            std::uniform_int_distribution<size_t> uniform_dist(MIN_BOUND, max_bound_);
            return uniform_dist(gen);
        }

        // 1. Roll a percentage within our precomputed limits
        std::uniform_real_distribution<double> uniform_dist(cdf_min_, cdf_max_);
        double p = uniform_dist(gen);

        // 2. Continuous conversion and rounding
        double sample = normal_inverse_cdf(p);
        size_t rounded = static_cast<size_t>(std::round(sample));

        // 3. Ultra-fast safety clamping
        return std::clamp(rounded, MIN_BOUND, max_bound_);
    }
};

}

#endif