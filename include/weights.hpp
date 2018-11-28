#ifndef WEIGHTS
#define WEIGHTS

#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <fstream>
#include "neuron.hpp"
#include "synapse.hpp"

using namespace std;

class Weights {
public:
	vector<Neuron*> output_neurons;

	vector<double> weights;
	vector<double> gradients;

	map<Neuron*, map<Neuron*, int> > map_layer_one;

	Weights(vector<Neuron*> output_neurons) : output_neurons(output_neurons) {}

	Weights() {}

	void set_weight_maps(Neuron* neuron);

	void helper_function(vector<Neuron*> neurons);

	void init() {
		helper_function(output_neurons);
		gradients.resize(weights.size());
		for (unsigned int i = 0 ; i < weights.size(); i ++) {
			gradients[i] = 0;
		}
	}

	int get_index(Neuron* n, Neuron* m);

	double get_weight(Neuron* n, Neuron* m);

	double get_gradient(Neuron* n, Neuron* m);

	void set_gradient(Neuron* n, Neuron* m, double gradient);

	void apply_gradient();
};

#endif