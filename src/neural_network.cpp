#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>

#include <fstream>
#include "neural_network.hpp"

using namespace std;

void Neural_network::init(int number_of_layers,
		int number_of_neurons_in_layer, int number_of_inputs, int number_of_outputs) {
	generate_network(number_of_layers,
		number_of_neurons_in_layer, number_of_inputs, number_of_outputs);
	Weights weights_(output_neurons);
	weights = weights_;
	weights.init();
}

void Neural_network::forward_propagation() {
	forward_propagation_helper(input_neurons);
	cout << "output is = ";
	for (unsigned int i = 0; i < output_neurons.size(); i++) {
		cout << output_neurons[i]->get_output() << endl;
	}
}

void Neural_network::forward_propagation_helper(vector<Neuron*> neurons) {
	for (unsigned int i = 0; i < neurons.size(); i++) {
		vector<Neuron*> parents = neurons[i]->get_parents();
		//cout << "parents size = " << parents.size() << endl;
		if (parents.size() > 0 ) {
			double sum = 0;
			for (unsigned int j = 0; j< parents.size(); j++) {
				double weight = weights.get_weight(parents[j], neurons[i]);
				sum += weight * parents[j]->get_output();
			}
			neurons[i]->init(sum);
		}
	}
	//cout << "node values are: ";
	for (unsigned int i = 0; i < neurons.size(); i++) {
	//	cout << neurons[i]->get_output() << " ";
	}
	//cout << endl;
	if (neurons[0]->get_children().size() > 0) {
		forward_propagation_helper(neurons[0]->get_children());
	}
}

void Neural_network::set_inputs(vector<double> inputs) {
	if (inputs.size() != input_neurons.size()) {
		cout << "Number of inputs is not correct" << endl;
	}

	for (unsigned int i = 0; i < input_neurons.size(); i++) {
		input_neurons[i]->set_input_node(inputs[i]);
	}
}

void Neural_network::train(vector<double> inputs, vector<double> targets, int iterations) {
	set_inputs(inputs);
	for (int i = 0; i < iterations ; i++) {
		//cout << "beginning forward propagation" << endl;
		forward_propagation_helper(input_neurons);
		back_propagation(targets);
	}
}

double Neural_network::delta(Neuron* neuron) {
	double output = neuron->get_output();
	if (neuron->is_output) {
		return (output - neuron->get_target()) * output * (1 - output);
	} else {
		double delta = 0;
		for (unsigned int i = 0 ; i < neuron->get_children().size(); i++) {
			//delta += weights[make_pair(neuron, neuron.children[i])]
			//	* neuron.children[i].get_delta()
			//	* neuron.get_output() * (1 - neuron.get_output());
		}
		return delta;
	}
}

int Neural_network::counter() {
	count++;
	return count;
}

void Neural_network::generate_network(int number_of_layers,
	int number_of_neurons_in_layer, int number_of_inputs, int number_of_outputs) {
	neurons.clear();
	input_neurons.clear();
	for (int i = 0; i < number_of_inputs; i++) {
		Neuron* neuron = new Neuron(counter());
		neuron->is_input = true;
		neurons.push_back(neuron);
		input_neurons.push_back(neuron);
	}

	vector<Neuron*> previous_layer = input_neurons;

	for (int i = 0 ; i < number_of_layers; i++) {
		// create next layer
		vector<Neuron*> hidden_layer_neurons;

		for (int j = 0 ; j < number_of_neurons_in_layer; j++) {
			//Neuron neuron(counter());
			Neuron* neuron = new Neuron(counter());
			neuron->set_parents(previous_layer);
			neurons.push_back(neuron);
			hidden_layer_neurons.push_back(neuron);
		}
		for (unsigned int j = 0 ; j < previous_layer.size(); j++) {
			previous_layer[j]->set_children(hidden_layer_neurons);
		}
		previous_layer = hidden_layer_neurons;
	}

	// deal with output layer

	for (int i = 0 ; i < number_of_outputs; i++) {
		//Neuron neuron(counter());
		Neuron* neuron = new Neuron(counter());
		neuron->is_output = true;
		neuron->set_parents(previous_layer);
		neurons.push_back(neuron);
		output_neurons.push_back(neuron);
	}

	for (int i = 0 ; i < number_of_neurons_in_layer; i++) {
		previous_layer[i]->set_children(output_neurons);
	}
}

void Neural_network::clean_up() {
	for (unsigned int i = 0; i < neurons.size(); i++) {
		delete neurons[i];
	}
}

void Neural_network::back_propagation_helper(vector<Neuron*> neurons) {
	for (unsigned int i = 0 ; i < neurons.size(); i++) {
		vector<Neuron*> children = neurons[i]->get_children();
		double sum = 0;
		for (unsigned int j = 0 ; j < children.size(); j++) {
			double weight_ij = weights.get_weight(children[j], neurons[i]);
			double delta_j = children[j]->get_delta();
			sum += weight_ij * delta_j;
		}
		double output = neurons[i]->get_output();
		neurons[i]->set_delta(sum * output * (1 - output));
	}
	if (neurons[0]->get_parents().size() > 0) {
		back_propagation_helper(neurons[0]->get_parents());
	}
}

void Neural_network::set_gradients(vector<Neuron*> neurons) {
	for (unsigned int i = 0 ; i < neurons.size(); i++) {
		vector<Neuron*> children = neurons[i]->get_children();
		for (unsigned int j = 0 ; j < children.size(); j++) {
			double delta_j = children[j]->get_delta();
			double output_i = (neurons[i]->get_output());
			double gradient_ij = output_i * delta_j;
			weights.set_gradient(children[j], neurons[i], gradient_ij);
		}
	}
	if (neurons[0]->get_parents().size() > 0) {
		set_gradients(neurons[0]->get_parents());
	}
}

void Neural_network::back_propagation(vector<double> targets) {
	// start from output neuron and work backworks computing the required delta
	if (targets.size() != output_neurons.size()) {
		cout << "Number of outputs is not correct" << endl;
	}

	// set value of delta for output neurons based on targets
	for (unsigned int i = 0 ; i < output_neurons.size(); i++) {
		double output = output_neurons[i]->get_output();
		double delta = (output - targets[i]) * output * (1 - output);
		output_neurons[i] -> set_delta(delta);
	}

	// set the value of delta for each neuron
	back_propagation_helper(output_neurons[0]->get_parents());

	// now compute the gradient of the weights

	set_gradients(output_neurons[0]->get_parents());
}

void Neural_network::apply_gradients() {
	weights.apply_gradient();
}