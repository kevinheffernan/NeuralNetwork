/********************** NOTES TO PADRAIG ************************
 *
 * 1) removed all spaces from the train/test files attached
 *    1 0 1 0 1 0 ... -> 101010 ...
 *
 * 2) each step (1-6) commented corresponds to the steps you gave
 *    in the suggested backpropagation algorithm
 *
 * 3) sorry if my code structure isn't great. this is my first
 *    Java program and was learning as I went along (I wanted to
 *    try your suggested implementation)
 ****************************************************************/

// using random utility to generate random bias + weights for each
// perceptron

import java.io.*;
import java.util.Arrays;
import java.util.Random;

// generating variables for use by all classes
// perceptron array: 'p' and corresponding results: 'outputs'
// 'loop_c' is simply a loop counter to record training time
// 'fullyTrained' is the stopping condition while training.
// outer and inner weight flags correspond to checking for weight
// changes. inner flags count weight changes for an array of 7
// perceptrons on each training set (x_i). if no weight change was
// recorded across all inner weight flags for a particular training
// case, then the outer flag for that particular case records no change.
// a 0 flag means a weight change and 1 means no change

abstract class staticVariables {
    
    /*************************
     network settings
     *************************/
    //setting 0 < α ≤ 1
    static float learningRate = 0.3f;
    //setting max and min numbers for weights
    static float minWeight = -0.05f;
    static float maxWeight =  0.05f;
    //setting max and min numbers for bias
    static float minBias = -1.0f;
    static float maxBias = -3.0f;
    
    static final int numPerceptrons = 7, numFonts = 3, numAtts = 63;
    static boolean fullyTrained = false;
    static perceptron [] p = new perceptron[numPerceptrons];
    static int [] outerWeightFlags = new int[numPerceptrons * numFonts];
    static int [] innerWeightFlags = new int[numPerceptrons];
    static int loop_c = 0;
    static int [] outputs = new int[numPerceptrons];
}

// the java random function generates a random number between
// 0.0 - 1.0 so I can limit the range by multipling the random
// number by ( max - min ) + min

class perceptron extends staticVariables {
    //vector weights
    double [] weights = new double[numAtts];
    //bias value
    double bias;
    //perceptron input
    double y_in;
    //perceptron output
    int y;
    
    perceptron() { this.initialise(); }
    
    public void initialise() {
        Random rand = new Random();
        
        for(int i = 0 ; i < numAtts ; i++) {
            //assign random number for weight
            weights[i] = rand.nextFloat() * (maxWeight - minWeight) + minWeight;
        }
        
        //assign random number for bias
        bias = rand.nextFloat() * (maxBias - minBias) + minBias;
    }
}

// the FileData class simply reads data from the train/test files accordingly.
// the first 63 values of each line are the attribute values and so are fed to
// each neuron as inputs. the remaining 7 values are sent as the class value for
// that particular case

class FileData extends staticVariables {
    //21 inputs : 70 total attributes (63 letter ; 7 class)
    int trainingData [][] = new int[numPerceptrons * numFonts][numPerceptrons + numAtts];
    int testData     [][] = new int[numFonts][numPerceptrons + numAtts];
    //perceptron input x, class attribute t (x : t)
    int [] x = new int[numAtts];
    int [] t = new int[numPerceptrons];
    
    int testCase = 0;
    String line;
    
    //when initialising, read all necessary data
    FileData() throws IOException { this.readData(); }
    
    void readData() throws IOException {
        BufferedReader fp = new BufferedReader(new FileReader("ocr_train.txt"));
        while ((line = fp.readLine()) != null) {
            for(int z = 0 ; z < line.length() ; z++) {
                trainingData[testCase][z] = Integer.parseInt(line.substring(z,z+1));
            }
            testCase++;
        }
        BufferedReader fp2 = new BufferedReader(new FileReader("ocr_test.txt"));
        testCase = 0;
        while ((line = fp2.readLine()) != null) {
            for(int z = 0 ; z < line.length() ; z++) {
                testData[testCase][z] = Integer.parseInt(line.substring(z,z+1));
            }
            testCase++;
        }
        fp2.close(); fp.close();
    }
    //if type 0 was sent, get training data. else test data
    //x_i, t_i = s_i : t_i
    void getAttributesAndClass(int index,int type) {
        if(type==0) {
            System.arraycopy(this.trainingData[index], 0, this.x, 0, numAtts);
            this.t = Arrays.copyOfRange(this.trainingData[index], numAtts, numPerceptrons + numAtts);
        }
        else {
            System.arraycopy(this.testData[index], 0, this.x, 0, numAtts);
            this.t = Arrays.copyOfRange(this.testData[index], numAtts, numPerceptrons + numAtts);
        }
    }
}


class Train extends staticVariables {
    void trainingLoopMethod(FileData data) {
        for (int index=0;index<(numPerceptrons*numFonts);index++) {
            
            ///////////////////////////////////
            // step 3
            ///////////////////////////////////
            data.getAttributesAndClass(index,0);
            
            ///////////////////////////////////
            // step 4
            ///////////////////////////////////
            for (int j = 0, summation = 0 ; j < numPerceptrons ; j++) {
                //∑ x(ij) * w(ij) : calculate summatiom
                for (int i=0; i < numAtts ; summation += data.x[i] * p[j].weights[i], i++);
                p[j].y_in = p[j].bias + summation;
                //calculate perceptron output
                p[j].y = p[j].y_in >= 0 ? 1 : -1;
                //re-calibrate 1|0 -> 1|-1
                data.t[j] = data.t[j]==0 ? -1 : 1;
                
                ///////////////////////////////////
                // step 5
                ///////////////////////////////////
                if (p[j].y != data.t[j]) {
                    for(int i=0; i < numAtts ; i++) {
                        //w(ij) = α * t(j) * x(i)
                        p[j].weights[i] += ((p[j].learningRate) * (data.t[j]) * (data.x[i]));
                    }
                    p[j].bias += (p[j].learningRate *  data.t[j]);
                    //if new weight != old weight : record weight change
                    innerWeightFlags[j] = 0;
                }
                //if old weight == new weight : record no weight change
                else { innerWeightFlags[j] = 1; }
                
                //reset summation value
                summation = 0;
            }
            
            //flag to test equality of weight flags
            boolean equal = false;
            
            ///////////////////////////////////
            // step 6
            ///////////////////////////////////
            for (int j = 0 ; j < numPerceptrons ; j++) {
                if   (innerWeightFlags[j] != 1) { equal = false; break; }
                else { equal = true; }
            }
            //if no weight change in instance, record no weight change
            outerWeightFlags[index] = (equal) ? 1 : 0;
            
            for (int j = 0 ; j < numPerceptrons*numFonts ; j++) {
                if   (outerWeightFlags[j] != 1) { equal = false; break; }
                else { equal = true; }
            }
            if (equal == true) { fullyTrained = true; break; }
        }
    }
    void printResults() {
        for (int index=0;index<(numPerceptrons);index++) {
            System.out.println(Arrays.toString(p[index].weights));
            System.out.println(p[index].bias);
        }
    }
}

class Test extends staticVariables {
    void testingLoopMethod(FileData data) {
        for (int index = 0 ; index < numFonts ; index++) {
            data.getAttributesAndClass(index,1);
            for (int j = 0, summation = 0 ; j < numPerceptrons ; j++) {
                
                for (int i=0; i < numAtts ; summation += data.x[i]*p[j].weights[i],i++);
                
                p[j].y_in = p[j].bias + summation;
                p[j].y = p[j].y_in >= 0 ? 1 : -1;
                outputs[j] = p[j].y;
                
                //reset summation value
                summation = 0;
            }
            
            int sum = 0;
            
            //important this is not initialised to 0.0d
            //setting as value of first perceptron
            double max = p[0].y_in;
            
            for(int c = 0 ; c < numPerceptrons ; sum += p[c].y, c++);
            
            //if more than one neuron fired, "winner takes all"
            if (sum == -7 || sum > -5 ) {
                System.out.println("needed to decide a winner");
                //reset outputs along the way
                for(int c = 0,winner = 0 ; c < numPerceptrons; c++) {
                    p[c].y = -1 ; outputs[c] = -1;
                    
                    if (p[c].y_in > max) { max = p[c].y_in; winner = c; }
                    if (c == 6) { p[winner].y = 1; outputs[winner] = 1; break;}
                }
            }
            System.out.println(Arrays.toString(outputs));
        }
    }
}

class main extends staticVariables {
    public static void main(String args[]) throws IOException {
        
        System.out.println("\n\tNeural Network (41450)\n");
        System.out.println("Learning rate: "+learningRate);
        
        ///////////////////////////////////
        // Step 0 : initialise perceptrons
        ///////////////////////////////////
        for (int i=0; i<numPerceptrons ; p[i]=new perceptron(),i++);
        
        //read in file data
        FileData data = new FileData();
        
        //get new training + testing classes
        Train training = new Train();
        Test testing = new Test();
        
        ///////////////////////////////////
        // step 1
        ///////////////////////////////////
        while(fullyTrained == false) {
            loop_c++;
            
            ///////////////////////////////////
            // step 2
            ///////////////////////////////////
            training.trainingLoopMethod(data);
	    }
        System.out.println("Training loops: "+loop_c+"\n");
        System.out.println("Results: \n");
        
        //testing
        testing.testingLoopMethod(data);
        
        System.out.println("");
    }
}