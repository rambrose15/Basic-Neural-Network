#include "header.h"
#include "layer.cpp"

class Brain{
    public:
        int layerNum;
        int inputNum;
        int outputNum;
        int nodeDensity;
        vector<Layer> layersList;
        Brain(int lsize, int isize, int osize, int nnum){
            SetVars(lsize, isize, osize, nnum);
            for (int i = 0; i < layerNum; i++){
                int sz, nxt;
                if (i == 0) sz = inputNum;
                else if (i == layerNum-1) sz = outputNum;
                else sz = nodeDensity;
                if (i < layerNum-2) nxt = nodeDensity;
                else if (i == layerNum-2) nxt = outputNum;
                else nxt = 0;
                layersList.push_back(*(new Layer(sz, nxt))); 
            }
        }
        vector<double> Compute(vector<double> inputs){
            for (int i = 0; i < layerNum; i++){
                inputs = layersList[i].Compute(inputs);
            }
            return inputs;
        }
        void Optimize(vector<pair<vector<double>,vector<double>>> &trainingCases){
            for (int layer = 0; layer < layerNum-1; layer++){ //For each layer of nodes
                for (int nodeIn = -1; nodeIn < layersList[layer].layerSize; nodeIn++){ // For each node in the layer, -1 is for bias
                    for (int nodeOut = 0; nodeOut < layersList[layer].nextLayerSize; nodeOut++){ // For each weight in the node/ bias in next layer
                        double partial = 0;
                        for (int i = 0, n = trainingCases.size(); i < n; i++){
                            partial += ErrorPartial(Compute(trainingCases[i].first), trainingCases[i].second, layer, nodeIn, nodeOut);
                        }
                        partial /= (double) trainingCases.size();
                        if (nodeIn == -1) layersList[layer+1].UpdateBias(-partial, nodeOut);
                        else layersList[layer].UpdateWeight(-partial, nodeIn, nodeOut);
                    } // Note: Currently recomputes training case output
                }
            }
        }
    private:
        void SetVars(int lsize, int isize, int osize, int nnum){
            layerNum = lsize;
            inputNum = isize;
            outputNum = osize;
            nodeDensity = nnum;
        }
        double ErrorPartial(vector<double> actualOutput, vector<double> desiredOutput, int parameterLayer, int nodeIn, int nodeOut){
            int n = actualOutput.size();
            double partial = 0;
            for (int i = 0; i < n; i++){
                if (nodeIn == -1) partial += 2 * (actualOutput[i] - desiredOutput[i]) * NodePartialB(actualOutput[i], layerNum-1, i, parameterLayer, nodeOut);
                else partial += 2 * (actualOutput[i] - desiredOutput[i]) * NodePartialW(actualOutput[i], layerNum-1, i, parameterLayer, nodeIn, nodeOut);
            }
            return partial;
        }
        double NodePartialW(double nodeVal, int nodeLayer, int nodeIndex, int weightLayer, int nodeIn, int nodeOut){
            if (nodeLayer == weightLayer + 1){
                if (nodeOut == nodeIndex){
                    return ActivationPrime(nodeVal - layersList[weightLayer].NodeVal(nodeIn) * layersList[weightLayer].NodeWeight(nodeIn, nodeOut))
                            * layersList[weightLayer].NodeVal(nodeIn);
                }
                else return 0;
            }
            else{
                double d = ActivationPrime(nodeVal - layersList[weightLayer].NodeVal(nodeIn) * layersList[weightLayer].NodeWeight(nodeIn, nodeOut))
                            * layersList[weightLayer].NodeVal(nodeIn);
                double s = 0;
                for (int i = 0; i < layersList[nodeLayer - 1].layerSize; i++){
                    s += NodePartialW(layersList[nodeLayer-1].NodeVal(i), nodeLayer-1, i, weightLayer, nodeIn, nodeOut) 
                        * layersList[nodeLayer-1].NodeWeight(i, nodeIndex);
                }
                return d * s;
            }
        }
        double NodePartialB(double outputVal, int nodeLayer, int nodeIndex, int biasLayer, int biasNode){
            if (biasLayer == nodeLayer){
                if (biasNode != nodeIndex) return 0;
                else{
                    return ActivationPrime(outputVal);
                }
            }
            else{
                double s = 0;
                for (int i = 0; i < layersList[nodeLayer - 1].layerSize; i++){
                    s += NodePartialB(layersList[nodeLayer-1].NodeVal(i), nodeLayer-1, i, biasLayer, biasNode) 
                    * layersList[nodeLayer-1].NodeWeight(i, nodeIndex);
                }
                return s * ActivationPrime(outputVal);
            }
        }
        double ActivationPrime(double x){
            return exp(-x) / pow(1 + exp(-x), 2);
        }
};