
/* This file is part of SOMHunter.
 *
 * Copyright (C) 2020 František Mejzlík <frankmejzlik@gmail.com>
 *                    Mirek Kratochvil <exa.exa@gmail.com>
 *                    Patrik Veselý <prtrikvesely@gmail.com>
 *
 * SOMHunter is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 2 of the License, or (at your option)
 * any later version.
 *
 * SOMHunter is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * SOMHunter. If not, see <https://www.gnu.org/licenses/>.
 */

#include "distfs.h"

#include <random>
#include <thread>
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <iostream>

#define SOM_ITERS 30000
//#define SOM_ITERS 100000
#define PARALLEL 2
//#define PARALLEL std::thread::hardware_concurrency()

// uncomment to get euclidean distances
#define EUCL

#ifdef EUCL
#define DIST_FUNC d_sqeucl
#define UNDIST_FUNC sqrtf
#else
#define DIST_FUNC d_manhattan
#define UNDIST_FUNC
#endif

// this helps with debugging floating-point overflows and similar nastiness,
// uncomment if needed.
//#define DEBUG_CRASH_ON_FPE

#ifdef DEBUG_CRASH_ON_FPE
#include <fenv.h>
#endif

using namespace std;

// some small numbers first!
static constexpr float min_boost = 0.00001f; // lower limit for the parameter

// this is added before normalizing the distances
static constexpr float zero_avoidance = 0.00000001f;

// a tiny epsilon for preventing singularities
static constexpr float koho_gravity = 0.00000001f;

struct dist_id
{
    float dist;
    size_t id;
};

static inline void
hswap(dist_id &a, dist_id &b)
{
    dist_id c = a;
    a = b;
    b = c;
}

static void
heap_down(dist_id *heap, size_t start, size_t lim)
{
    for (;;)
    {
        size_t L = 2 * start + 1;
        size_t R = L + 1;
        if (R < lim)
        {
            float dl = heap[L].dist;
            float dr = heap[R].dist;

            if (dl > dr)
            {
                if (heap[start].dist >= dl)
                    break;
                hswap(heap[L], heap[start]);
                start = L;
            }
            else
            {
                if (heap[start].dist >= dr)
                    break;
                hswap(heap[R], heap[start]);
                start = R;
            }
        }
        else if (L < lim)
        {
            if (heap[start].dist < heap[L].dist)
                hswap(heap[L], heap[start]);
            break; // exit safely!
        }
        else
            break;
    }
}

void som(size_t n,
         size_t k,
         size_t dim,
         size_t niter,
         const float *const points,
         std::vector<float> &koho,
         const std::vector<float> &nhbrdist,
         const float alphasA[2],
         const float radiiA[2],
         const float alphasB[2],
         const float radiiB[2],
         const float *const scores,
         std::mt19937 &rng)
{
    std::discrete_distribution<size_t> random(scores, scores + n);

    float thresholdA0 = radiiA[0];
    float alphaA0 = alphasA[0];
    float thresholdADiff = radiiA[1] - radiiA[0];
    float alphaADiff = alphasA[1] - alphasA[0];
    float thresholdB0 = radiiB[0];
    float alphaB0 = alphasB[0];
    float thresholdBDiff = radiiB[1] - radiiB[0];
    float alphaBDiff = alphasB[1] - alphasB[0];

    for (size_t iter = 0; iter < niter; ++iter)
    {
        size_t point = random(rng);
        float riter = iter / float(niter);

        size_t nearest = 0;
        {
            float nearestd = DIST_FUNC(
                points + dim * point, koho.data(), dim);
            for (size_t i = 1; i < k; ++i)
            {
                float tmp =
                    DIST_FUNC(points + dim * point,
                              koho.data() + dim * i,
                              dim);
                if (tmp < nearestd)
                {
                    nearest = i;
                    nearestd = tmp;
                }
            }
        }

        float thresholdA = thresholdA0 + riter * thresholdADiff;
        float thresholdB = thresholdB0 + riter * thresholdBDiff;
        float alphaA = alphaA0 + riter * alphaADiff;
        float alphaB = alphaB0 + riter * alphaBDiff;

        for (size_t i = 0; i < k; ++i)
        {
            float d = nhbrdist[i + k * nearest];

            float alpha;

            if (d > thresholdA)
            {
                if (d > thresholdB)
                    continue;
                alpha = alphaB;
            }
            else
                alpha = alphaA;

            for (size_t j = 0; j < dim; ++j)
                koho[j + i * dim] +=
                    alpha *
                    (points[j + point * dim] - koho[j + i * dim]);
        }
    }
}

/* this serves for classification into small clusters */
void mapPointsToKohos(size_t start,
                      size_t end,
                      size_t k,
                      size_t dim,
                      const float *const &points,
                      const std::vector<float> &koho,
                      std::vector<size_t> &mapping)
{
    for (size_t point = start; point < end; ++point)
    {
        size_t nearest = 0;
        float nearestd =
            DIST_FUNC(points + dim * point, koho.data(), dim);
        for (size_t i = 1; i < k; ++i)
        {
            float tmp = DIST_FUNC(points + dim * point,
                                  koho.data() + dim * i,
                                  dim);
            if (tmp < nearestd)
            {
                nearest = i;
                nearestd = tmp;
            }
        }

        mapping[point] = nearest;
    }
}

size_t weighted_example(const std::vector<size_t> &subset, const float *const scores)
{
    std::vector<float> fs(subset.size());
    for (size_t i = 0; i < subset.size(); ++i)
        fs[i] = scores[subset[i]];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<size_t> dist(fs.begin(), fs.end());
    return subset[dist(gen)];
}

size_t nearest_cluster_with_atleast(
    const float *const vec,
    std::unordered_map<size_t, size_t> &stolen_count,
    const std::vector<std::vector<size_t>> &mapping,
    const std::vector<float> &koho,
    const size_t dim)
{
    float min = std::numeric_limits<float>::max();
    size_t res = 0;
    for (size_t i = 0; i < mapping.size(); ++i)
    {
        if (mapping[i].size() > stolen_count[i])
        {
            float tmp =
                d_sqeucl(koho.data() + dim * i,
                         vec,
                         dim);
            if (min > tmp)
            {
                min = tmp;
                res = i;
            }
        }
    }

    return res;
}

void som_display(const float *const points,
                 const float *const scores,
                 const size_t n,
                 const size_t dim,
                 const size_t swidth,
                 const size_t sheight,
                 size_t *output)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    // Prepare cached distences
    std::vector<float> nhbrdist(
        swidth * swidth *
        sheight * sheight);
    for (size_t x1 = 0; x1 < swidth; ++x1)
        for (size_t y1 = 0; y1 < sheight; ++y1)
            for (size_t x2 = 0; x2 < swidth;
                 ++x2)
                for (size_t y2 = 0;
                     y2 < sheight;
                     ++y2)
                    nhbrdist
                        [x1 +
                         swidth *
                             (y1 +
                              sheight *
                                  (x2 +
                                   swidth *
                                       y2))] =
                            abs(float(x1) - float(x2)) +
                            abs(float(y1) - float(y2));

    // Prepare hyperparameters
    std::vector<float> koho(swidth *
                                sheight *
                                dim,
                            0);
    float negAlpha = -0.01f;
    float negRadius = 1.1f;
    float alphasA[2] = {0.3f, 0.1f};
    float alphasB[2] = {negAlpha * alphasA[0],
                        negAlpha * alphasA[1]};
    float radiiA[2] = {float(swidth +
                             sheight) /
                           3,
                       0.1f};
    float radiiB[2] = {negRadius * radiiA[0],
                       negRadius * radiiA[1]};
    // Compute SOM
    som(n,
        swidth * sheight,
        dim,
        SOM_ITERS,
        points,
        koho,
        nhbrdist,
        alphasA,
        radiiA,
        alphasB,
        radiiB,
        scores,
        rng);

    // Map values to clusters in parallel
    std::vector<size_t> mapping_per_image(n);
    {
#ifdef PARALLEL
        size_t n_threads = PARALLEL;
        std::vector<std::thread> threads(n_threads);

        auto worker = [&](size_t id)
        {
            size_t start = id * n / n_threads;
            size_t end = (id + 1) * n / n_threads;
            mapPointsToKohos(start,
                             end,
                             swidth *
                                 sheight,
                             dim,
                             points,
                             koho,
                             mapping_per_image);
        };

        for (size_t i = 0; i < n_threads; ++i)
            threads[i] = std::thread(worker, i);

        for (size_t i = 0; i < n_threads; ++i)
            threads[i].join();
#else
        mapPointsToKohos(0,
                         n,
                         swidth *
                             sheight,
                         dim,
                         points,
                         koho,
                         mapping_per_image);
#endif
    }

    // Reverse mapping
    std::vector<std::vector<size_t>> mapping_per_cluster;
    mapping_per_cluster.resize(swidth * sheight);
    for (size_t im = 0; im < mapping_per_image.size(); ++im)
        mapping_per_cluster[mapping_per_image[im]].push_back(im);

    // Sample from mapping
    std::vector<size_t> ids;
    ids.resize(swidth * sheight);

    for (size_t i = 0; i < swidth; ++i)
    {
        for (size_t j = 0; j < sheight; ++j)
        {
            if (mapping_per_cluster[i + swidth * j]
                    .empty())
            {
                ids[i + swidth * j] =
                    -1;
            }
            else
            {
                size_t id = weighted_example(
                    mapping_per_cluster[i + swidth * j], scores);
                ids[i + swidth * j] = id;
            }
        }
    }

    // Steal representants for empty clusters
    std::unordered_map<size_t, size_t> stolen_count;
    for (size_t i = 0; i < swidth * sheight;
         ++i)
    {
        stolen_count.emplace(i, 1);
    }

    for (size_t i = 0; i < swidth; ++i)
    {
        for (size_t j = 0; j < sheight; ++j)
        {
            if (mapping_per_cluster[i + swidth * j]
                    .empty())
            {
                auto k = koho[(i + swidth * j) * dim];

                size_t clust = nearest_cluster_with_atleast(&k, stolen_count, mapping_per_cluster, koho, dim);

                stolen_count[clust]++;
                std::vector<size_t> ci = mapping_per_cluster[clust];

                for (size_t ii : ids)
                {
                    auto fi =
                        std::find(ci.begin(), ci.end(), ii);
                    if (fi != ci.end())
                        ci.erase(fi);
                }

                assert(!ci.empty());

                size_t id = weighted_example(ci, scores);
                ids[i + swidth * j] = id;
            }
        }
    }

    // Copy result
    for (size_t i = 0; i < swidth * sheight; ++i)
    {
        output[i] = ids[i];
    }
}
