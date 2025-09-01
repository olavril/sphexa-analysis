#pragma once

#include "cstone/findneighbors.hpp"

using cstone::LocalIndex;

template<class T>
T updateH(unsigned ng0, unsigned nc, T h)
{
    constexpr T c0  = 1023.0;
    constexpr T exp = 1.0 / 10.0;
    return h * T(0.5) * std::pow(T(1) + c0 * ng0 / T(nc), exp);
}

template<class T, class KeyType>
void findNeighborsSph(const T* x, const T* y, const T* z, T* h, LocalIndex firstId, LocalIndex lastId,
                      const cstone::Box<T>& box, const cstone::OctreeNsView<T, KeyType>& treeView, unsigned ng0,
                      unsigned ngmax, LocalIndex* neighbors, unsigned* nc)
{
    LocalIndex numWork = lastId - firstId;

    unsigned ngmin = ng0 / 4;

    size_t        numFails     = 0;
    constexpr int maxIteration = 10;

#pragma omp parallel for reduction(+ : numFails)
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id    = i + firstId;
        unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);

        int iteration = 0;
        while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
        {
            h[id] = updateH(ng0, ncSph, h[id]);
            ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
        }
        numFails += (iteration >= maxIteration);

        nc[i] = ncSph;
    }

    if (numFails)
    {
        std::cout << "Coupled h-neighbor count updated failed to converge for " << numFails << " particles"
                  << std::endl;
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, d.ng0,
                     d.ngmax, d.neighbors.data(), d.nc.data() + startIndex);
}

template<class T>
void resizeNeighbors(std::vector<T>& vector, size_t size)
{
    double growthRate       = 1.05;
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}
