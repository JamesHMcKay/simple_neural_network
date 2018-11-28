#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <map>
#include <fstream>
#include "neuron.hpp"
#include "synapse.hpp"
#include "weights.hpp"

using namespace std;

class Neural_network {
private:

	Weights weights;

	vector<Neuron*> input_neurons;
	vector<Neuron*> output_neurons;

	vector<Neuron*> neurons;

	int number_of_neurons;

	int count = 0;


public:
	Neural_network() {};

	void init(int number_of_layers,
		int number_of_neurons_in_layer, int number_of_inputs, int number_of_outputs);

	void train(vector<double> inputs, vector<double> targets, int iterations);

	void forward_propagation_helper(vector<Neuron*> neurons);

	void back_propagation(vector<double> targets);

	double delta(Neuron* neuron);

	void clean_up();

	int counter();

	void back_propagation_helper(vector<Neuron*> neurons);

	void set_gradients(vector<Neuron*> neurons);

	void generate_network(int number_of_layers,
		int number_of_neurons_in_layer, int number_of_inputs, int number_of_outputs);

	void forward_propagation_helper(vector<double> inputs);
	
	void forward_propagation();

	void set_inputs(vector<double> inputs);

	void apply_gradients();
};
#endif