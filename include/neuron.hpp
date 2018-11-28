#ifndef NEURON
#define NEURON

#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class Neuron {
private:
	double output;

public:
	int id = 0;
	bool is_output = false;
	bool is_input = false;
	double target;
	double delta;

	vector<Neuron*> children;
	vector<Neuron*> parents;

	Neuron(int id) : id (id) {}

	Neuron() {}

	void set_target(double target_in) {
		target = target_in;
		is_output = true;
	}

	void set_parents(vector<Neuron*> parents_input) {
		parents = parents_input;
	}

	void init(double sum) {
		output = sigmoud_function(sum);
	}

	vector<Neuron*> get_parents() {
		return parents;
	}

	vector<Neuron*> get_children() {
		return children;
	}

	void set_children(vector<Neuron*> children_input) {
		children = children_input;
	}

	double sigmoud_function(double value) {
		return 1.0 / (1 + exp(-value));
	}

	double get_output() {
		return output;
	}

	void set_input_node(double input) {
		output = input;
	}

	double get_delta() {
		return delta;
	}

	void set_delta(double value) {
		delta = value;
		//cout << "delta for neuron " << id << " set to = " << delta << endl;
	}

	double get_target() {
		if (is_output) {
			cout << "get target";
			return target;
		} else {
			cout << "attempted to access target value for a non-output node";
			return 0;
		}
	}
};
#endif