//standard libraries
#include <iostream>
#include <sstream>

//custom includes
#include "dataEntry.h"
#include "dataReader.h"
#include "neuralNetwork.h"

//use standard namespace
using namespace std;

void main()
{	

#pragma region IRIS
	//create data set reader
	dataReader dreader;

	//load data file
	dreader.loadDataFile("iris.dat", 4, 3);
	dreader.setCreationApproach(STATIC);

	//create neural network
	// input node number=4
	// output node number=3
	// number of hidden layer=3
	// 1th hidden layer node number=100
	// 2th hidden layer node number=90
	// 3th hidden layer node number=80
	neuralNetwork nn(4, 3, 3, 100, 90, 80);
	nn.enableLogging("trainingResults.csv");
	nn.setLearningParameters(0.01, 0.8);//Setting the learning rate and momentum rate as (lr,m)
	nn.setDesiredAccuracy(100);
	nn.setMaxEpochs(500);

	//dataset
	dataSet* dset;

	for (int i = 0; i < dreader.nDataSets(); i++)
	{
		dset = dreader.getDataSet();
		nn.trainNetwork(dset->trainingSet, dset->generalizationSet, dset->validationSet);
	}

	cout << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
#pragma endregion


}
