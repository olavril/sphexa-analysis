#pragma once 

// correct periodic conditions for a computed distance in 1 dimension
// input:   distance in ONE(!) dimension
// output:  distance in this direction corrected for periodic conditions
double correct_periodic_cond(double dist){
    if (dist > 0.5){
        dist -= 1.0;
    } else if (dist < -0.5){
        dist += 1.0;
    }
    return dist;
}

// correct Rank or Layer for periodic dondition
// input:   number of the Rank/Layer you want to correct
//          total number of ranks/Layers
// output:  corrected number of Rank/Layer
int correct_periodic_cond(int Rank, const int NumRanks){
    if (Rank >= NumRanks){
        Rank -= NumRanks;
    } else if (Rank < 0){
        Rank += NumRanks;
    }
    return Rank;
}