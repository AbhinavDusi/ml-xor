#include <vector>
#include <random>
#include <iostream>

#include "NeuralNet.hpp"

int main() {
    std::vector<int> topology(4); 
    topology.push_back(2); 
    topology.push_back(4); 
    topology.push_back(1); 

    NeuralNet net(topology); 

    std::mt19937 rng(time(nullptr));

    for (int i = 0; i < 2000; i++) {
        int a = (rng()/(double) rng.max())%2; 
        int b = (rng()/(double) rng.max())%2; 

        std::vector<double> input(2); 
        input.push_back(a); 
        input.push_back(b); 

        std::vector<double> target(1); 
        target.push_back(a^b); 

        net.feed_forward(input);
        net.back_prop(target); 
    }

    int a = (rng()/(double) rng.max())%2; 
    int b = (rng()/(double) rng.max())%2;

    std::vector<double> input(2); 
    input.push_back(a); 
    input.push_back(b); 

    std::vector<double> result = net.get_result(); 

    std::cout << a << " XOR " << b << " = " result.front() << std::endl; 

    return 0; 
}