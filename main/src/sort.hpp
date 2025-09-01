#pragma once

/*
* This script provides an implementation of the introsort algorithm.
*/

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <stdint.h>
#include <omp.h>

template<typename sort_t, class... dragVector_t>
void swap(const uint64_t ind1, const uint64_t ind2, std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const uint64_t iBuffer)
{   
    sort_t buffer = sortVector[ind1];
    sortVector[ind1] = sortVector[ind2];
    sortVector[ind2] = buffer;
    std::apply([ind1,ind2,iBuffer](auto&&... args) {((args[iBuffer] = args[ind1]), ...);},    dragVectors);
    std::apply([ind1,ind2,iBuffer](auto&&... args) {((args[ind1] = args[ind2]), ...);},       dragVectors);
    std::apply([ind1,ind2,iBuffer](auto&&... args) {((args[ind2] = args[iBuffer]), ...);},    dragVectors);
    return;
}

// insertionsort algorithm applied to the [startIndex,stopIndex] interval of the vectors
// both startIndex & stopIndex are INCLUDED in this interval
template<typename sort_t, class... dragVector_t>
void insertionSort(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const uint64_t startIndex, const uint64_t stopIndex, const uint64_t iBuffer)
{
    // loop over the section
    for (size_t i = startIndex; i <= stopIndex; i++){
        // loop over entries left of the current value that is to be sorted
        for (size_t j = i; j > startIndex; j--){
            // if the sortVector value to the left is larger than the current value, swap them 
            if (sortVector[j-1] > sortVector[j]){
                swap(j-1,j,sortVector,dragVectors,iBuffer);
            }
            // if not, end the inner loop and move on
            else {break;}
        }
    }
}

// function to identify the child nodes of a given entry in a heap starting from startIndex in the data
std::tuple<int,int> identifyChildren(const int i, const int startIndex)
{
    std::tuple<int,int> Children;
    int leftChild;
    int rightChild;

    if (i == 0){
        leftChild  = startIndex + 1;
        rightChild = startIndex + 2;
    } else {
        leftChild  = startIndex + 2*(i+1)-1;
        rightChild = startIndex + 2*(i+1);
    }
    return std::make_tuple(leftChild,rightChild);
}

// function to ensure the max heap property on a [startIndex,stopIndex] interval of the data
// both startIndex & stopIndex are INCLUDED in this interval
template<typename sort_t, class... dragVector_t>
void heapify(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int i, const int startIndex, const int stopIndex, const uint64_t iBuffer)
{
    int max = startIndex + i;
    std::tuple<int,int> Children = identifyChildren(i, startIndex);
    const int leftChild  = std::get<0>(Children);
    const int rightChild = std::get<1>(Children);

    //check left child
    if (leftChild <= stopIndex){
        if (sortVector[leftChild] > sortVector[max]){
            max = leftChild;
        }
    }
    if (rightChild <= stopIndex){
        if (sortVector[rightChild] > sortVector[max]){
            max = rightChild;
        }
    }

    //swap elements if necessary
    if (max != startIndex+i){
        swap(startIndex+i, max, sortVector, dragVectors, iBuffer);
        heapify(sortVector,dragVectors,max-startIndex,startIndex, stopIndex, iBuffer);
    }
    return ;
}

// function to build a max heap on a [startIndex,stopIndex] interval of the data
template<typename sort_t, class... dragVector_t>
void buildMaxHeap(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int startIndex, const int stopIndex, const uint64_t iBuffer)
{   
    //get full size of the vector interval - we want to build the full heap
    const int SizeUnsorted = stopIndex - startIndex +1;

    for (int i = (int)(SizeUnsorted/2)-1; i >= 0; i--){
        heapify(sortVector, dragVectors, i, startIndex, stopIndex, iBuffer);
    }
    return ;
}

//implementation of the heapsort algorithm on a [startIndex,stopIndex] interval of the data
template<typename sort_t, class... dragVector_t>
void heapSort(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int startIndex, int stopIndex, const uint64_t iBuffer)
{
    // create max-Heap
    buildMaxHeap(sortVector, dragVectors, startIndex, stopIndex, iBuffer);

    for (int i = stopIndex; i > startIndex; i--){
        swap(i, startIndex, sortVector,dragVectors,iBuffer);
        stopIndex--;
        heapify(sortVector,dragVectors, 0, startIndex, stopIndex, iBuffer);
    }

    return;
}

// function to select a suitable pivot index for partitioning using the median-of-3 approach
template<typename sort_t>
int selectPivot(std::vector<sort_t>& sortVector, const int startIndex, const int stopIndex)
{
    const int middleIndex = (int)((startIndex+stopIndex)/2);
    int PivotIndex = -1;

    if (((sortVector[startIndex] < sortVector[middleIndex]) && (sortVector[middleIndex] < sortVector[stopIndex])) or 
        ((sortVector[stopIndex] < sortVector[middleIndex]) && (sortVector[middleIndex] < sortVector[startIndex])))
    {
        PivotIndex = middleIndex;
    }
    else if (((sortVector[middleIndex] < sortVector[startIndex]) && (sortVector[startIndex] < sortVector[stopIndex])) or 
            ((sortVector[stopIndex] < sortVector[startIndex]) && (sortVector[startIndex] < sortVector[middleIndex])))
    {
        PivotIndex = startIndex;
    }
    else 
    {
        PivotIndex = stopIndex;
    }

    return PivotIndex;
}

//function to partition a [startIndex,stopIndex] interval of the data using a seleted pivot index
template<typename sort_t, class... dragVector_t>
int partition(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int startIndex, const int stopIndex, const uint64_t iBuffer)
{   
    //select pivot element
    int PivotIndex = selectPivot(sortVector, startIndex, stopIndex);
    if (stopIndex != PivotIndex) {
        swap(stopIndex,PivotIndex, sortVector,dragVectors, iBuffer);
        PivotIndex = stopIndex;
    }
    int Leftwall = startIndex;

    for (int i = startIndex; i < stopIndex; i++)
    {
        if (sortVector[i] < sortVector[PivotIndex]){
            swap(i, Leftwall, sortVector,dragVectors, iBuffer);
            Leftwall++;
        }
    }
    swap(Leftwall,PivotIndex, sortVector,dragVectors, iBuffer);

    return Leftwall;
}

// implementation of the introsort recursion
// selects the sorting algorithm to be usedon the given data section [startIndex,stopIndex]
template<typename sort_t, class... dragVector_t>
void introsortRecursion(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int startIndex, const int stopIndex, 
                        uint64_t maxdepth, const uint64_t iBuffer)
{
    // compute lenth of the section that should be sorted
    // startIndex & stopIndex are both part of the section! hence the +1
    int length = stopIndex - startIndex + 1;

    // decide what sorting algorithm to use depending on the size of 
    // the section & the maxdepth for the recursion
    if (length == 0)
    {
        return;
    }
    else if (length < 16)
    {   
        //if the section is small enough, insertionsort is the most efficient
        insertionSort(sortVector,dragVectors, startIndex,stopIndex, iBuffer);
    } 
    else if (maxdepth == 0) 
    {   
        // if the maximum depth was reached use heapsort to mitigate the worst case of quicksort
        heapSort(sortVector,dragVectors, startIndex,stopIndex, iBuffer);
    }
    else
    {   
        // otherwise do partition with pivot element similar to quicksort
        int PivotIndex = partition(sortVector,dragVectors, startIndex, stopIndex, iBuffer);

        //recursion similar to quicksort
        introsortRecursion(sortVector,dragVectors, startIndex, PivotIndex, maxdepth-1, iBuffer);
        introsortRecursion(sortVector,dragVectors, PivotIndex+1, stopIndex, maxdepth-1, iBuffer);
    }

    return;
}

template<typename sort_t, class... dragVector_t>
void split(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, std::vector<uint64_t>& firstElements, std::vector<uint64_t>& lastElements, 
           const uint64_t iBuffer, int step, const int numProcessors)
{   
    // find the largest partition in the current setup
    int maxIndex = 0;
    uint64_t maxLength  = lastElements[maxIndex] - firstElements[maxIndex];
    for (size_t i = 1; i <= step; i++){
        if ((lastElements[i] - firstElements[i]) > maxLength){
            maxIndex = i;
            maxLength = lastElements[maxIndex] - firstElements[maxIndex];
        }
    }
    // partition this largest section
    int PivotIndex = partition(sortVector,dragVectors, firstElements[maxIndex], lastElements[maxIndex], iBuffer);
    // move back later firstElements entries
    
    for (size_t i = firstElements.size()-2; i > maxIndex; i--){
        firstElements[i+1] = firstElements[i];
    }
    for (int i = lastElements.size()-2; i >= maxIndex; i--){
        lastElements[i+1] = lastElements[i];
    }
    firstElements[maxIndex+1]   = PivotIndex+1;
    lastElements[maxIndex]      = PivotIndex;

    // recursion
    step++;
    if (step == numProcessors-1){
        return;
    } else {
        split(sortVector,dragVectors,firstElements,lastElements,iBuffer,step,numProcessors);
        return;
    }
}


// function to invert the order of the vectors
// this function is used when invertFlag is set and you want the result to go from large to small
template<typename sort_t, class... dragVector_t>
void invert(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const uint64_t iBuffer)
{
    for (size_t i = 0; i < sortVector.size()/2; i++){
        swap(i,sortVector.size()-1-i, sortVector,dragVectors, iBuffer);
    }
    return;
}

// function to sort vectors
// input:   sortVector is the vector which is relevant for sorting (IDs, number of comms, etc.)
//          dragVectors are the vectors to chich the same sorting operations should be applied (create with std::tie(vec1,vec2,...))
//          numProcessors is the number of processors that should be used for multiprocessing
//          invertFlag can be set if you want to sort from largest to smallest (true). Default is smallest to largest (false)
// output:  None, the input vectors are modified according to the operations
template<typename sort_t, class... dragVector_t>
void sort(std::vector<sort_t>& sortVector, std::tuple<dragVector_t&...> dragVectors, const int numProcessors, const bool invertFlag = false)
{
    // resize the drag Vectors to create buffer space for switch operations
    std::apply([numProcessors](auto&&... args) {((args.resize(args.size() + numProcessors)), ...);}, dragVectors);

    // compute max depth for the quicksort recursion
    uint64_t maxdepth = static_cast<uint64_t>(std::ceil(std::log2(sortVector.size()) * 2));

    // buffewr for inverting & sorting if numProcessors == 1
    uint64_t iBuffer = static_cast<uint64_t>(sortVector.size());

    // sorting the vectors
    if (numProcessors > 1){
        std::vector<uint64_t> firstElements(numProcessors, 0);
        std::vector<uint64_t> lastElements(numProcessors, sortVector.size()-1);
        split(sortVector,dragVectors,firstElements,lastElements,iBuffer,0,numProcessors);
        #pragma omp parallel for
        for (size_t iProc = 0; iProc < numProcessors; iProc++){
            introsortRecursion(sortVector,dragVectors, firstElements[iProc],lastElements[iProc],maxdepth,iBuffer);
        }
    } else {
        introsortRecursion(sortVector,dragVectors, 0,sortVector.size()-1,maxdepth,iBuffer);
    }

    // invert the vectors to go from large to small instead of small to large if the flag was set
    if (invertFlag){
        invert(sortVector,dragVectors, iBuffer);
    }

    // reduce the size of the drag vectors again to remove buffer storage
    std::apply([numProcessors](auto&&... args) {((args.resize(args.size() - numProcessors)), ...);}, dragVectors);

    return;
}
