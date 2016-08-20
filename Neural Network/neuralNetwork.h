#ifndef NNETWORK
#define NNETWORK

//standard libraries
#include <math.h>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>

//custom includes
#include "dataEntry.h"
#include<stdarg.h>

using namespace std;

//Constant Defaults!
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  

class neuralNetwork
{

//private members
//----------------------------------------------------------------------------------------------------------------
private:

#pragma region Properties
	//learning parameters
	double learningRate;					// adjusts the step size of the weight update	
	double momentum;						// improves performance of stochastic learning (don't use for batch)

	//number of neurons
	int nInputNode, nOutputNode;
	int* nHiddenNode;

	//number of hidden layer
	int nHiddenLayer;

	//neurons
	double* inputNeurons;
	double** hiddenNeurons;
	double* outputNeurons;

	//weights
	double** wInputHidden;
	double**** wHiddenHidden;
	double** wHiddenOutput;

	//epoch counter
	long epoch;
	long maxEpochs;

	//accuracy required
	double desiredAccuracy;

	//change to weights
	double** deltaInputHidden;
	double**** deltaHiddenHidden;
	double** deltaHiddenOutput;

	//error gradients
	double** hiddenErrorGradients;
	double* outputErrorGradients;

	//accuracy stats per epoch
	double trainingSetAccuracy;
	double validationSetAccuracy;
	double generalizationSetAccuracy;
	double trainingSetMSE;
	double validationSetMSE;
	double generalizationSetMSE;

	//batch learning flag
	bool useBatch;

	//log file handle
	bool logResults;
	fstream logFile;
	int logResolution;
	int lastEpochLogged;

#pragma endregion

//public methods
//----------------------------------------------------------------------------------------------------------------
public:

	//constructor
	neuralNetwork(int in, int out, int hiddenLayerNumber, ...) : nInputNode(in), nOutputNode(out), nHiddenLayer(hiddenLayerNumber), epoch(0), logResults(false), logResolution(10), lastEpochLogged(-10), trainingSetAccuracy(0), validationSetAccuracy(0), generalizationSetAccuracy(0), trainingSetMSE(0), validationSetMSE(0), generalizationSetMSE(0)
	{
		//create neuron lists
		//--------------------------------------------------------------------------------------------------------
		//Input Layer Nodes
		inputNeurons = new(double[in + 1]);
		for (int i = 0; i < in; i++) inputNeurons[i] = 0;

		//create bias neuron
		inputNeurons[in] = -1;

		//Hidden Layer nodes
		nHiddenNode = new(int[hiddenLayerNumber]);
		va_list tmp;
		va_start(tmp, hiddenLayerNumber);

		nHiddenNode[0] = va_arg(tmp, int);
		for (int i = 1; i < hiddenLayerNumber; i++) {
			nHiddenNode[i] = va_arg(tmp, int);
		}

		hiddenNeurons = new (double*[hiddenLayerNumber]);
		for (int i = 0; i < hiddenLayerNumber; i++)
		{
			hiddenNeurons[i] = new (double[nHiddenNode[i]]);
			for (int k = 0; k < nHiddenNode[i]; k++) hiddenNeurons[i][k] = 0;
		}
		//create bias neuron
		for (int i = 0; i < hiddenLayerNumber; i++)
			hiddenNeurons[i][nHiddenNode[i]] = -1;

		//Output Layer nodes
		outputNeurons = new(double[out]);
		for (int i = 0; i < out; i++) outputNeurons[i] = 0;

		//create weight lists (include bias neuron weights)
		//--------------------------------------------------------------------------------------------------------
		//weight of the edge between input and hidden layers
		wInputHidden = new(double*[in + 1]);
		for (int i = 0; i <= in; i++)
		{
			wInputHidden[i] = new (double[nHiddenNode[0]]);
			for (int j = 0; j < nHiddenNode[0]; j++) wInputHidden[i][j] = 0;
		}
		//weight of the edge between former and latter hidden layers
		wHiddenHidden = new(double***[hiddenLayerNumber]);
		for (int f = 0; f < hiddenLayerNumber - 1; f++)//former layer
		{
			wHiddenHidden[f] = new(double**[hiddenLayerNumber]);
			for (int l = 1; l < hiddenLayerNumber; l++)//latter layer
			{
				wHiddenHidden[f][l] = new(double*[nHiddenNode[f]]);
				for (int i = 0; i <= nHiddenNode[f]; i++)//former layer node index 
				{
					wHiddenHidden[f][l][i] = new (double[nHiddenNode[l]]);
					for (int j = 0; j < nHiddenNode[l]; j++)//latter layer node index 
						wHiddenHidden[f][l][i][j] = 0;
				}
			}
		}
		//weight of the edge between hidden and output layers
		wHiddenOutput = new(double*[nHiddenNode[hiddenLayerNumber - 1] + 1]);
		for (int i = 0; i <= nHiddenNode[hiddenLayerNumber - 1]; i++)
		{
			wHiddenOutput[i] = new (double[out]);
			for (int j = 0; j < out; j++) wHiddenOutput[i][j] = 0;
		}

		//create delta lists
		//--------------------------------------------------------------------------------------------------------
		//delta list of input layer to first hidden layer node
		deltaInputHidden = new(double*[in + 1]);
		for (int i = 0; i <= in; i++)
		{
			deltaInputHidden[i] = new (double[nHiddenNode[0]]);
			for (int j = 0; j < nHiddenNode[0]; j++) deltaInputHidden[i][j] = 0;
		}
		//delta list of hidden layers node
		deltaHiddenHidden = new(double***[nHiddenLayer + 1]);
		for (int f = 0; f < hiddenLayerNumber - 1; f++)//former layer
		{
			deltaHiddenHidden[f] = new(double**[hiddenLayerNumber]);
			for (int l = 1; l < hiddenLayerNumber; l++)//latter layer
			{
				deltaHiddenHidden[f][l] = new(double*[nHiddenNode[f]]);
				for (int i = 0; i <= nHiddenNode[f]; i++)//former layer node index 
				{
					deltaHiddenHidden[f][l][i] = new (double[nHiddenNode[l]]);
					for (int j = 0; j < out; j++) deltaHiddenHidden[f][l][i][j] = 0;
				}
			}
		}
	

		//delta list of last hidden to output layer
		deltaHiddenOutput = new( double*[nHiddenNode[nHiddenLayer-1] + 1] );
		for ( int i=0; i <= nHiddenNode[nHiddenLayer-1]; i++ ) 
		{
			deltaHiddenOutput[i] = new (double[out]);			
			for ( int j=0; j < out; j++ ) deltaHiddenOutput[i][j] = 0;		
		}

		//create error gradient storage
		//--------------------------------------------------------------------------------------------------------
		//hidden layers nodes error gradient
		hiddenErrorGradients = new (double*[hiddenLayerNumber]);
		for (int h = 0; h < hiddenLayerNumber; h++)//former layer
		{
			hiddenErrorGradients[h] = new(double[nHiddenNode[h] + 1]);
			for (int i = 0; i <= nHiddenNode[h]; i++) hiddenErrorGradients[h][i] = 0;
		}
		//output layer node error gradient
		outputErrorGradients = new( double[out + 1] );
		for ( int i=0; i <= out; i++ ) outputErrorGradients[i] = 0;
		
		//initialize weights
		//--------------------------------------------------------------------------------------------------------
		initializeWeights();

		//default learning parameters
		//--------------------------------------------------------------------------------------------------------
		learningRate = LEARNING_RATE; 
		momentum = MOMENTUM; 

		//use stochastic learning by default
		useBatch = false;
		
		//stop conditions
		//--------------------------------------------------------------------------------------------------------
		maxEpochs = MAX_EPOCHS;
		desiredAccuracy = DESIRED_ACCURACY;			
	}

	//destructor
	~neuralNetwork()
	{
		//delete neurons
		delete[] inputNeurons;
		for (int i = 0; i <= nHiddenNode[0]; i++) delete[] hiddenNeurons[i];
		delete[] hiddenNeurons;
		delete[] outputNeurons;

		//delete weight storage
		for (int i=0; i <= nInputNode; i++) delete[] wInputHidden[i];
		delete[] wInputHidden;

		for (int f = 0; f <= nHiddenLayer; f++)
		{
			for (int l = 0; l <= nHiddenLayer; l++)
			{
				for (int i = 0; i <= nHiddenNode[f]; i++)
					delete[] wHiddenHidden[f][l][i];
				delete[] wHiddenHidden[f][l];
			}
			delete[] wHiddenHidden[f];
		}
		delete[] wHiddenHidden;
		
		for (int j = 0; j <= nHiddenNode[nHiddenLayer-1]; j++) delete[] wHiddenOutput[j];
		delete[] wHiddenOutput;

		//delete delta storage
		for (int i=0; i <= nInputNode; i++) delete[] deltaInputHidden[i];
		delete[] deltaInputHidden;

		for (int f = 0; f <= nHiddenLayer; f++)
		{
			for (int l = 0; l <= nHiddenLayer; l++)
			{
				for (int i = 0; i <= nHiddenNode[f]; i++)
					delete[] deltaHiddenHidden[f][l][i];
				delete[] deltaHiddenHidden[f][l];
			}
			delete[] deltaHiddenHidden[f];
		}
		delete[] deltaHiddenHidden;

		for (int j=0; j <= nHiddenNode[nHiddenLayer-1]; j++) delete[] deltaHiddenOutput[j];
		delete[] deltaHiddenOutput;

		//delete error gradients
		for (int i = 0; i <= nHiddenLayer; i++) delete[] hiddenErrorGradients[i];
		delete[] hiddenErrorGradients;
		delete[] outputErrorGradients;

		//close log file
		if ( logFile.is_open() ) logFile.close();
	}

	//set learning parameters
	void setLearningParameters(double lr, double m)
	{
		learningRate = lr;		
		momentum = m;
	}

	//set max epoch
	void setMaxEpochs(int max)
	{
		maxEpochs = max;
	}

	//set desired accuracy 
	void setDesiredAccuracy(float d)
	{
		desiredAccuracy = d;
	}

	//enable batch learning
	void useBatchLearning()
	{
		useBatch = true;
	}

	//enable stochastic learning
	void useStochasticLearning()
	{
		useBatch = false;
	}

	//enable logging of training results
	void enableLogging(const char* filename, int resolution = 1)
	{
		//create log file 
		if ( ! logFile.is_open() )
		{
			logFile.open(filename, ios::out);

			if ( logFile.is_open() )
			{
				//write log file header
				logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
				
				//enable logging
				logResults = true;
				
				//resolution setting;
				logResolution = resolution;
				lastEpochLogged = -resolution;
			}
		}
	}

	//resets the neural network
	void resetWeights()
	{
		//reinitialize weights
		initializeWeights();		
	}

	//feed data through network
	double* feedInput( double* inputs)
	{
		//feed data into the network
		feedForward(inputs);

		//return results
		return outputNeurons;
	}

	//train the network
	void trainNetwork( vector<dataEntry*> trainingSet, vector<dataEntry*> generalizationSet, vector<dataEntry*> validationSet )
	{
		cout<< endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " Learning Rate: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
			<< " " << nInputNode << " Input Neurons, " << nHiddenNode << " Hidden Neurons, " << nOutputNode << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

		//reset epoch and log counters
		epoch = 0;
		lastEpochLogged = -logResolution;
			
		//train network using training dataset for training and generalization dataset for testing
		//--------------------------------------------------------------------------------------------------------
		while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )				
		{			
			//store previous accuracy
			double previousTAccuracy = trainingSetAccuracy;
			double previousGAccuracy = generalizationSetAccuracy;

			//use training set to train network
			runTrainingEpoch( trainingSet );

			//get generalization set accuracy and MSE
			generalizationSetAccuracy = getSetAccuracy( generalizationSet );
			generalizationSetMSE = getSetMSE( generalizationSet );

			//Log Training results
			if (logResults && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
			{
				logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
				lastEpochLogged = epoch;
			}
			
			//print out change in training /generalization accuracy (only if a change is greater than a percent)
			if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
			{	
				cout << "Epoch :" << epoch;
				cout << " TAcc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
				cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;				
			}
			
			//once training set is complete increment epoch
			epoch++;

		}//end while

		//get validation set accuracy and MSE
		validationSetAccuracy = getSetAccuracy(validationSet);
		validationSetMSE = getSetMSE(validationSet);

		//log end
		logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
		logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;
				
		//out validation accuracy and MSE
		cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
		cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
		cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
	}
	
	
//private methods
//----------------------------------------------------------------------------------------------------------------
private:

	//initialize weights and weight changes
	void initializeWeights()
	{
		//init random number generator
		srand( (unsigned int) time(0) );
			
		//set weights between input and hidden to a random value between -05 and 0.5
		//--------------------------------------------------------------------------------------------------------
		for(int i = 0; i <= nInputNode; i++)
		{		
			for(int j = 0; j < nHiddenNode[0]; j++) 
			{
				//set weights to random values
				wInputHidden[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

				//create blank delta
				deltaInputHidden[i][j] = 0;
			}
		}
		//set weights between hidden and hidden to a random value between -05 and 0.5
		//--------------------------------------------------------------------------------------------------------
		for (int f = 0; f < nHiddenLayer-1; f++)
		{
			for (int l = 1; l < nHiddenLayer; l++)
			{
				for (int i = 0; i < nHiddenNode[f]; i++)
				{
					for (int j = 0; j < nHiddenNode[l]; j++)
					{
						//set weights to random values
						wHiddenHidden[f][l][i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

						//create blank delta
						deltaHiddenHidden[f][l][i][j] = 0;
					}
				}
			}
		}

		//set weights between hidden and output to a random value between -05 and 0.5
		//--------------------------------------------------------------------------------------------------------
		for(int i = 0; i <= nHiddenNode[nHiddenLayer-1]; i++)
		{		
			for(int j = 0; j < nOutputNode; j++) 
			{
				//set weights to random values
				wHiddenOutput[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

				//create blank delta
				deltaHiddenOutput[i][j] = 0;
			}
		}
	}

	//run a single training epoch
	void runTrainingEpoch( vector<dataEntry*> trainingSet )
	{
		//incorrect patterns
		double incorrectPatterns = 0;
		double mse = 0;
			
		//for every training pattern
		for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
		{						
			//feed inputs through network and backpropagate errors
			feedForward( trainingSet[tp]->pattern );
			backpropagate( trainingSet[tp]->target );	

			//pattern correct flag
			bool patternCorrect = true;

			//check all outputs from neural network against desired values
			for ( int k = 0; k < nOutputNode; k++ )
			{					
				////pattern incorrect if desired and output differ
				//if ( getRoundedOutputValue( outputNeurons[k] ) != trainingSet[tp]->target[k] ) patternCorrect = false;
				//
				//calculate MSE
				mse += pow((outputNeurons[k] - trainingSet[tp]->target[k]), 2);
			}
			//
			double o1 = outputNeurons[0]; double t1 = trainingSet[tp]->target[0];
			double o2 = outputNeurons[1]; double t2 = trainingSet[tp]->target[1];
			double o3 = outputNeurons[2]; double t3 = trainingSet[tp]->target[2];
			if ((o1 > o2 && o1 > o3) && (t1<1 ))
				patternCorrect = false;
			if ((o2 > o1 && o2 > o3) && (t2 <1))
				patternCorrect = false;
			if ( (o3 > o1 && o3 > o1) &&  (t3 <1))
				patternCorrect = false;
			//if pattern is incorrect add to incorrect count
			if ( !patternCorrect ) incorrectPatterns++;	
			
		}//end for

		//if using batch learning - update the weights
		if ( useBatch ) updateWeights();
		
		//update training accuracy and MSE
		trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
		trainingSetMSE = mse / ( nOutputNode * trainingSet.size() );
	}

	//feed input forward
	void feedForward( double *inputs )
	{
		//set input neurons to input values
		for(int i = 0; i < nInputNode; i++) inputNeurons[i] = inputs[i];
		
		//Calculate First Hidden Layer values - include bias neuron
			//--------------------------------------------------------------------------------------------------------
			for (int j = 0; j < nHiddenNode[0]; j++)
			{
				//clear value
				hiddenNeurons[0][j] = 0;

				//get weighted sum of inputs and bias neuron
				for (int i = 0; i <= nInputNode; i++) hiddenNeurons[0][j] += inputNeurons[i] * wInputHidden[i][j];

				//set to result of sigmoid
				hiddenNeurons[0][j] = activationFunction(hiddenNeurons[0][j]);
			}
			//Calculate Remained Hidden Layer values - include bias neuron
			//--------------------------------------------------------------------------------------------------------
			for (int h = 1; h < nHiddenLayer; h++)
			{//For Each Hidden Layer
				for (int j = 0; j < nHiddenNode[h]; j++)
				{
					//clear value
					hiddenNeurons[h][j] = 0;

					//get weighted sum of inputs and bias neuron
					for (int i = 0; i <= nHiddenNode[h - 1]; i++) hiddenNeurons[h][j] += hiddenNeurons[h - 1][i] * wHiddenHidden[h-1][h][i][j];

					//set to result of sigmoid
					hiddenNeurons[h][j] = activationFunction(hiddenNeurons[h][j]);
				}

			}

		//Calculating Output Layer values - include bias neuron
		//--------------------------------------------------------------------------------------------------------
		for(int k=0; k < nOutputNode; k++)
		{
			//clear value
			outputNeurons[k] = 0;				
			
			//get weighted sum of inputs and bias neuron
			for( int j=0; j <= nHiddenNode[nHiddenLayer-1]; j++ ) outputNeurons[k] += hiddenNeurons[nHiddenLayer-1][j] * wHiddenOutput[j][k];
			
			//set to result of sigmoid
			outputNeurons[k] = activationFunction( outputNeurons[k] );
		}
	}

	//modify weights according to output
	void backpropagate( double* desiredValues )
	{		
		//modify deltas between hidden and output layers
		//--------------------------------------------------------------------------------------------------------
		for (int k = 0; k < nOutputNode; k++)
		{
			//get error gradient for every output node
			// error at the output neurons (Desired value – actual value) and multiplying it by the gradient of the sigmoid function
			outputErrorGradients[k] = getOutputErrorGradient(desiredValues[k], outputNeurons[k]);

			//for all nodes in the last hidden layer and bias neuron
			for (int j = 0; j <= nHiddenNode[nHiddenLayer - 1]; j++)
			{
				//calculate change in weight
				if (!useBatch) deltaHiddenOutput[j][k] = learningRate * hiddenNeurons[nHiddenLayer - 1][j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j][k];
				else deltaHiddenOutput[j][k] += learningRate * hiddenNeurons[nHiddenLayer - 1][j] * outputErrorGradients[k];
			}
		}
		//modify deltas between hidden and hidden layers
		//--------------------------------------------------------------------------------------------------------
		for (int h = nHiddenLayer - 1; h > 0;h--)
		{
			for (int j = 0; j < nHiddenNode[h]; j++)
			{
				//get error gradient for every hidden node
				hiddenErrorGradients[h][j] = getHiddenErrorGradient(h,j);

				//for all nodes in former hidden layer and bias neuron
				for (int i = 0; i <= nHiddenNode[h-1]; i++)
				{
					//calculate change in weight // since  inputNeurons[i] * hiddenErrorGradients[j] is the diffretiation of cost function with respect to
					//weights of hidden-->output
					if (!useBatch) deltaHiddenHidden[h - 1][h][i][j] = learningRate * hiddenNeurons[h-1][i] * hiddenErrorGradients[h ][j] + momentum * deltaHiddenHidden[h - 1][h][i][j];
					else deltaHiddenHidden[h - 1][h][i][j] += learningRate * hiddenNeurons[h-1][i] * hiddenErrorGradients[h ][j];

				}
			}

		}
		//modify deltas between input and hidden layers
		//--------------------------------------------------------------------------------------------------------
		for (int j = 0; j < nHiddenNode[0]; j++)
		{
			//get error gradient for every hidden node
			hiddenErrorGradients[0][j] = getHiddenErrorGradient( 0,j );

			//for all nodes in input layer and bias neuron
			for (int i = 0; i <= nInputNode; i++)
			{
				//calculate change in weight // scince  inputNeurons[i] * hiddenErrorGradients[j] is the diffretiation of cost function with respect to
				//weights of hidden-->output
				if ( !useBatch ) deltaInputHidden[i][j] = learningRate * inputNeurons[i] * hiddenErrorGradients[0][j] + momentum * deltaInputHidden[i][j];
				else deltaInputHidden[i][j] += learningRate * inputNeurons[i] * hiddenErrorGradients[0][j]; 

			}
		}
		
		//if using stochastic learning update the weights immediately
		if ( !useBatch ) updateWeights();
	}

	//update weights
	void updateWeights()
	{
		//input -> hidden weights
		//--------------------------------------------------------------------------------------------------------
		for (int i = 0; i <= nInputNode; i++)
		{
			for (int j = 0; j < nHiddenNode[0]; j++) 
			{
				//update weight
				wInputHidden[i][j] += deltaInputHidden[i][j];	
				
				//clear delta only if using batch (previous delta is needed for momentum
				if (useBatch) deltaInputHidden[i][j] = 0;				
			}
		}
		
		//hidden -> hidden weights
		//--------------------------------------------------------------------------------------------------------
		for (int f = 0; f < nHiddenLayer-1; f++)//former hidden layer
		{
			for (int l = 1; l < nHiddenLayer; l++)//later hidden layer
			{
				for (int i = 0; i <= nHiddenNode[f]; i++)
				{
					for (int j = 0; j < nHiddenNode[l]; j++)
					{
						//update weight
						wHiddenHidden[f][l][i][j] += deltaHiddenHidden[f][l][i][j];

						//clear delta only if using batch (previous delta is needed for momentum
						if (useBatch) deltaHiddenHidden[f][l][i][j] = 0;
					}
				}
			}
		}
		//hidden -> output weights
		//--------------------------------------------------------------------------------------------------------
		for (int j = 0; j <= nHiddenNode[nHiddenLayer-1]; j++)
		{
			for (int k = 0; k < nOutputNode; k++) 
			{					
				//update weight
				wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
				
				//clear delta only if using batch (previous delta is needed for momentum)
				if (useBatch)deltaHiddenOutput[j][k] = 0;
			}
		}
	}

	//activation function
	inline double activationFunction( double x )
	{
		//sigmoid function
		return 1/(1+exp(-x));
	}

	//get error gradient for ouput layer
	//∇E(w) = ( ∂E/∂w1, ...,∂E/∂wM)
	//The vector of partial derivatives is called the gradient of the error
	inline double getOutputErrorGradient(double desiredValue, double outputValue)
	{
		//return error gradient
		return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
	}

	//get error gradient for hidden layer
	double getHiddenErrorGradient(int h, int j )
	{
		//get sum of hidden->output weights * output error gradients
		double weightedSum = 0;
		if (h==nHiddenLayer-1)
			for( int k = 0; k < nOutputNode; k++ ) weightedSum += wHiddenOutput[j][k] * outputErrorGradients[k];
		else
			for (int k = 0; k < nHiddenNode[h + 1]; k++) weightedSum += wHiddenHidden[h][h+1][j][k] * hiddenErrorGradients[h+1][k];
		//return error gradient
		return hiddenNeurons[h][j] * ( 1 - hiddenNeurons[h][j] ) * weightedSum;
	}

	//round up value to get a boolean result
	int getRoundedOutputValue( double x )
	{
		if ( x < 0.1 ) return 0;
		else if ( x > 0.9 ) return 1;
		else return -1;
	}	
	//feed forward set of patterns and return error
	double getSetAccuracy( vector<dataEntry*> set )
	{
		double incorrectResults = 0;
		
		//for every training input array
		for ( int tp = 0; tp < (int) set.size(); tp++)
		{						
			//feed inputs through network and backpropagate errors
			feedForward( set[tp]->pattern );
			
			//correct pattern flag
			bool correctResult = true;

			//check all outputs against desired output values
			//for ( int k = 0; k < nOutput; k++ )
			//{					
			//	//set flag to false if desired and output differ
			//	if ( getRoundedOutputValue(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;
			//}
			double o1 = outputNeurons[0]; double t1 = set[tp]->target[0];
			double o2 = outputNeurons[1]; double t2 = set[tp]->target[1];
			double o3 = outputNeurons[2]; double t3 = set[tp]->target[2];
			if ((o1 > o2 && o1 > o3) && (t1<1))//ifo1=1 and o2!=1 and o3!=1 means training output is calss o1 where the desired output t1 is not 1 so its false output
				correctResult = false;
			if ((o2 > o1 && o2 > o3) && (t2 <1))
				correctResult = false;
			if ((o3 > o1 && o3 > o1) && (t3 < 1))
				correctResult = false;

			//inc training error for a incorrect result
			if ( !correctResult ) incorrectResults++;	
			
		}//end for
		
		//calculate error and return the true labeled percentage (100-incorrect percentage)
		return 100 - (incorrectResults/set.size() * 100);
	}

	//feed forward set of patterns and return MSE
	double getSetMSE ( vector<dataEntry*> set )
	{
		double mse = 0;
		
		//for every training input array
		for ( int tp = 0; tp < (int) set.size(); tp++)
		{						
			//feed inputs through network and backpropagate errors
			feedForward( set[tp]->pattern );
			
			//check all outputs against desired output values
			for ( int k = 0; k < nOutputNode; k++ )
			{					
				//sum all the MSEs together
				mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
			}		
			
		}//end for
		
		//calculate error and return as percentage
		return mse/(nOutputNode * set.size());
	}
};

#endif
