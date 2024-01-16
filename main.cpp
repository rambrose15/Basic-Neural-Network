#include <iostream>
#include "brain.cpp"
using namespace std;

int main(){
    Brain b = *(new Brain(3, 1, 1, 4)); // layernum, inputnum, outputnum, nodeDensity
    vector<pair<vector<double>,vector<double>>> cases;
    vector<double> in1 = {0.2}, out1 = {0.7}, in2 = {0.6}, out2 = {0.4};
    cases.push_back({in1, out1});
    cases.push_back({in2, out2});
    for (int i = 0; i < 1000; i++){
        b.Optimize(cases);
    }
    vector<double> test = {0.6};
    vector<double> after = b.Compute(test);
    for (auto x : after) cout << x << ", ";
    test = {0.2};
    after = b.Compute(test);
    for (auto x : after) cout << x << ", ";
}