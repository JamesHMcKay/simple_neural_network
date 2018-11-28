#ifndef SYNAPSE
#define SYNAPSE

#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>

#include <fstream>
#include "neuron.hpp"

using namespace std;

class Synapse {
private:
	Neuron* _start;
	Neuron* _end;

public:

	double _weight;

	Synapse(Neuron* start, Neuron* end, double weight) {
		_start = start;
		_end = end;
		_weight = weight;
	}

	Synapse() {}

};
#endif



