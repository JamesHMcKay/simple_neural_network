#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <fstream>
#include "weights.hpp"

using namespace std;

void Weights::set_weight_maps(Neuron* neuron) {
	vector<Neuron*> parents = neuron-> get_parents();
	map<Neuron*, int> map_layer_two;

	if (parents.size() > 0) {
		for (unsigned int i = 0; i < parents.size() ; i++) {
			double random_number = (double) rand() / (RAND_MAX);
			double second_random_number = (double) rand() / (RAND_MAX);
			if (second_random_number < 0.5) {
				random_number = -random_number;
			}
			weights.push_back(random_number);
			map_layer_two[parents[i]] = weights.size() - 1;
		}
	}

	map_layer_one[neuron] = map_layer_two;
}

void Weights::helper_function(vector<Neuron*> neurons) {
	if (neurons.size() > 0) {
		for (unsigned int i = 0; i < neurons.size() ; i++) {
			set_weight_maps(neurons[i]);
		}
		helper_function(neurons[0]->get_parents());
	}
}

int Weights::get_index(Neuron* n, Neuron* m) {
	int index = map_layer_one[n][m];
	int index_reverse = map_layer_one[m][n];

	if (index == index_reverse) {
		return index;
	} else if (index == 0) {
		return index_reverse;
	} else {
		return index;
	}
}

double Weights::get_weight(Neuron* n, Neuron* m) {
	return weights[get_index(n, m)];
}

void Weights::set_gradient(Neuron* n, Neuron* m, double gradient) {
	gradients[get_index(n, m)] += gradient;
}

void Weights::apply_gradient() {
	for (unsigned int i = 0; i < weights.size(); i++) {
		weights[i] = weights[i] - 2.0 * gradients[i];
		gradients[i] = 0;
	}
}