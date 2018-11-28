#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <complex>
#include <fstream>
#include "neural_network.hpp"
#include "neuron.hpp"
#include <ctime>
#include <iostream>
#include <string>

using namespace std;

class Test {
public:
	int value;
	Test(int value) : value(value) {}
};

double test_function(double a) {
	return 2*a;
}

int main() {

	Neural_network neural_network;
	neural_network.init(3, 7, 3, 1);

	vector<double> targets_1 = {1};
	vector<double> inputs_1 = {1,1,1};

	vector<double> targets_2 = {1};
	vector<double> inputs_2 = {0,0,0};

	vector<double> targets_3 = {0};
	vector<double> inputs_3 = {1,0,1};

	vector<double> targets_4 = {0};
	vector<double> inputs_4 = {0,1,0};

	vector<double> targets_5 = {0};
	vector<double> inputs_5 = {1,1,0};
	
	for (int i = 0; i < 100 ; i++) {
		neural_network.train(inputs_1, targets_1,1);
		neural_network.train(inputs_2, targets_2,1);
		neural_network.train(inputs_3, targets_3,1);
		neural_network.train(inputs_4, targets_4,1);
		neural_network.train(inputs_5, targets_5,1);
		neural_network.apply_gradients();
	}
	cout << "for 0, 1 ";
	neural_network.set_inputs({0,1,0});
	neural_network.forward_propagation();


	cout << "for 1, 0 ";
	neural_network.set_inputs({1,0,1});
	neural_network.forward_propagation();

	cout << "for 0, 0 ";
	neural_network.set_inputs({0,0,0});
	neural_network.forward_propagation();

	cout << "for 1, 1 ";
	neural_network.set_inputs({1,1,1});
	neural_network.forward_propagation();

	cout << "for 1, 1, 0 ";
	neural_network.set_inputs({1,1,0});
	neural_network.forward_propagation();

	neural_network.clean_up();

/*
	vector<Test> test_vector;
	vector<Test*> test_pointer_vector;

	for (int i = 0; i < 3; i++) {
		Test* test = new Test(i);
		test_vector.push_back(*test);
		test_pointer_vector.push_back(test);
	}


	cout << "values are : " << endl;
	for (int i = 0; i < 3; i++) {
		cout << " vector of values: " << test_vector[i].value << endl;
		cout << " vector of pointers: " << test_pointer_vector[i]->value << endl;
		delete test_pointer_vector[i];
	}
	*/

}
