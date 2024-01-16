#include "header.h"

class Node{
    public:
        Node(int connections){
            InitBias();
            InitWeights(connections);
        }
        vector<double> NodeOutput(double input){
            SetValue(input);
            vector<double> r(weights);
            for (int i = 0, n = r.size(); i < n; i++){
                r[i] *= value;
            }
            return r;
        }
        double GetValue(){
            return value;
        }
        double SetValue(double input){
            value = ActivationFunction(input + bias);
            return value;
        }
        double GetWeight(int weightIndex){
            return weights[weightIndex];
        }
        void UpdateWeight(double val, int nodeOut){
            weights[nodeOut] += val;
        }
        void UpdateBias(double val){
            bias += val;
        }
    private:
        double value;
        double bias;
        vector<double> weights;
        void InitBias(){
            srand(time(NULL));
            bias = ((double)(rand() % 200)) / 100.0 - 1.0; // Sets bias to real number between -1 and 1
        }
        void InitWeights(int sz){
            srand(time(NULL));
            for (int i = 0; i < sz; i++){
                weights.push_back(((double)(rand() % 200)) / 100.0 - 1.0); // Sets each weight to a real number between -1 and 1
            }
        }
        double ActivationFunction(double d){
            return 1.0 / (1.0 + exp(-d));
        }
};


class Layer{
    public:
        int layerSize;
        int nextLayerSize;
        vector<Node> nodes;
        Layer(int size, int nextSize){
            layerSize = size;
            nextLayerSize = nextSize;
            for (int i = 0; i < size; i++){
                nodes.push_back(*(new Node(nextSize)));
            }
        }

        vector<double> Compute(vector<double> inputs){
            vector<double> r (nextLayerSize, 0);
            for (int i = 0, n = inputs.size(); i < n; i++){
                if (nextLayerSize != 0) VectorAdd(r, nodes[i].NodeOutput(inputs[i]));
                else { // Last layer case, calculate values then give back
                    r.push_back(nodes[i].SetValue(inputs[i]));
                }
            }
            return r;
        }

        double NodeVal(int nodeIndex){
            return nodes[nodeIndex].GetValue();
        }
        double NodeWeight(int nodeIndex, int weightIndex){
            return nodes[nodeIndex].GetWeight(weightIndex);
        }

        void UpdateWeight(double val, int nodeIn, int nodeOut){
            nodes[nodeIn].UpdateWeight(val, nodeOut);
        }
        void UpdateBias(double val, int node){
            nodes[node].UpdateBias(val);
        }

        void VectorAdd(vector<double> &v1, vector<double> v2){
            for (int i = 0, n = v1.size(); i < n; i++){
                v1[i] += v2[i];
            }
        }
};